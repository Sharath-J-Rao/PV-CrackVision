import dotenv from 'dotenv';
import express from 'express';
import type { NextFunction, Request, Response } from 'express';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import JSZip from 'jszip';
import sharp from 'sharp';
import { detectCellBoxes, getCellDetectorStatus, setCellDetectorMode } from './server/cellDetector.ts';
import { listCellAnnotations, cellAnnotationImagePath, mergeImportedCellAnnotations, saveCellAnnotations, saveNegativeCellSample } from './server/cellAnnotationStore.ts';
import { scaleCellBoxesToSourceImage } from './server/cellAnnotationCoordinates.ts';
import { cancelCellYoloTraining, pauseCellYoloTraining, resumeCellYoloTraining, getCellTrainingLogPath, getCellTrainingMetricsCsv, getCellTrainingStatus, getCellTrainingCheckpointBuffer, resetCellTrainingDataAndModel, rollbackCellModel, setupCellTrainingEngine, startCellYoloTraining } from './server/cellTraining.ts';
import { analyzeInspectionImage } from './server/cvPipeline.ts';
import type { ReviewedCellInput } from './server/cellDetectionGeometry.ts';
import { loadMetadata, loadMetadataFile, metadataCount } from './server/metadataStore.ts';
import { addFeedbackMany, addFeedbackManyTransactional, undoFeedbackBatch, existingFeedbackForImage, clearFeedbackDataset, feedbackCounts, feedbackImagePath, getLastFeedbackUndo, initializeCellFeedbackStore, listFeedback, removeFeedback } from './server/feedbackStore.ts';
import { cancelClassifierTraining, classifierLogText, classifierStatusSnapshot, clearClassifierModelsAndData, isClassifierMode, startClassifierTraining } from './server/classifierTraining.ts';
import { DEFAULT_TRAINING_SETTINGS, getTrainingSettings, saveTrainingSettings } from './server/trainingSettings.ts';
import { learnProfileFromBoxes } from './server/cellTypeFilter.ts';
import type {
  GeoJsonFeatureCollection,
  GeoJsonPointFeature,
  InspectionConfig,
  InspectionResult,
} from './src/types/inspection.ts';
import { DATA_ROOT, migrateLegacyTrainingData } from './server/dataPaths.ts';
import { getLastRecoveryReport, recoverPreviousModels } from './server/modelRecovery.ts';
import { loadCellClassifier } from './server/cellClassifier.ts';
import { bufferedCellCrop } from './server/crackModel.ts';
import { browseFolders, dataFolderUsage, describeDataFolder, resetDataFolderToDefault, setDataFolder } from './server/dataFolder.ts';
import { importModelFile } from './server/modelImport.ts';
import { getModelHealth, resolveCellModel } from './server/modelResolver.ts';
import { isPackagedApp } from './server/appConfig.ts';
import { deleteReview, isValidReviewHash, loadReview, saveReview } from './server/reviewStore.ts';
import { createCellTrainingDataBundle, parseCellTrainingDataBundle } from './server/cellTrainingDataBundle.ts';
import { APP_VERSION } from './server/appVersion.ts';
import { isLoopbackAddress, validateRequestSecurity } from './server/requestSecurity.ts';

// .env.local first (per README), then .env
dotenv.config({ path: ['.env.local', '.env'] });

// Migrate legacy per-release runtime data before any request can read the shared stores.
const DATA_MIGRATION_RESULT = migrateLegacyTrainingData();
// v1.4.17.7: find and install trained models from earlier releases (every start; never overwrites a newer model).
recoverPreviousModels();
for (const engine of ['anomaly', 'classifier'] as const) void loadCellClassifier(engine).catch(() => undefined); // warm up: picks the GPU when available

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const MAX_ZIP_ENTRIES = 300;
const MAX_ZIP_UNPACKED_BYTES = 1.5 * 1024 * 1024 * 1024;

// v1.3.1+ uses reviewed cell crops; v1.4.1+ adds direct-cell YOLO training and an advanced-only annotation workspace.
// Legacy feedback schemas are preserved in timestamped backup folders during the one-time migration.
initializeCellFeedbackStore(); // legacy-schema migration only; the old boosted-tree/CNN model file is no longer used and is left untouched

// Tiny fixed-window limiter so the API can't be hammered 
const rawRateLimit = Number(process.env.RATE_LIMIT_PER_MIN);
const RATE_LIMIT_PER_MIN = Number.isFinite(rawRateLimit) ? Math.max(60, Math.min(3600, Math.floor(rawRateLimit))) : 600;
const hits = new Map<string, { n: number; reset: number }>();
function rateLimit(req: Request, res: Response, next: NextFunction): void {
  // v1.4.16.4: the bundled UI (same machine) polls status and fires many requests during batch detection; throttling it produced
  // "Too many requests" mid-batch. Cross-site and DNS-rebinding requests are already rejected by validateRequestSecurity, so only
  // non-loopback (remote-access) clients are rate limited.
  if (isLoopbackAddress(req.socket.remoteAddress ?? '')) return next();
  const now = Date.now();
  const key = req.ip || 'local';
  const h = hits.get(key);
  if (hits.size > 10000) {
    for (const [client, entry] of hits) if (now > entry.reset) hits.delete(client);
  }
  if (!h || now > h.reset) {
    hits.set(key, { n: 1, reset: now + 60_000 });
    return next();
  }
  if (++h.n > RATE_LIMIT_PER_MIN) {
    res.status(429).json({ error: 'Too many requests, slow down.' });
    return;
  }
  next();
}

function readConfigHeader(req: Request): unknown {
  const raw = req.headers['x-inspection-config'];
  if (typeof raw !== 'string' || !raw.trim()) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

function safeDecode(v: string): string {
  try {
    return decodeURIComponent(v);
  } catch {
    return v; // malformed %-escape: use raw name instead of 500
  }
}

function parseInspectionConfig(raw: unknown): InspectionConfig {
  const obj = (typeof raw === 'object' && raw !== null ? raw : {}) as Record<string, unknown>;
  const sens = obj.sensitivity;
  return {
    tileSizePx:
      typeof obj.tileSizePx === 'number' && Number.isFinite(obj.tileSizePx)
        ? obj.tileSizePx
        : 1360,
    overlapRatio:
      typeof obj.overlapRatio === 'number' && Number.isFinite(obj.overlapRatio)
        ? obj.overlapRatio
        : 0.18,
    sensitivity:
      sens === 'high' || sens === 'balanced' || sens === 'strict' ? sens : 'high',
    maxTilesToInspect:
      typeof obj.maxTilesToInspect === 'number' && Number.isFinite(obj.maxTilesToInspect)
        ? obj.maxTilesToInspect
        : 12,
    includeSurfaceResidue:
      typeof obj.includeSurfaceResidue === 'boolean' ? obj.includeSurfaceResidue : true,
    contrastEnhancement:
      typeof obj.contrastEnhancement === 'boolean' ? obj.contrastEnhancement : true,
    edgeFocusMode:
      typeof obj.edgeFocusMode === 'boolean' ? obj.edgeFocusMode : true,
    onlyImagesWithIssuesInGeoJson:
      typeof obj.onlyImagesWithIssuesInGeoJson === 'boolean' ? obj.onlyImagesWithIssuesInGeoJson : true,
    rejectGround: typeof obj.rejectGround === 'boolean' ? obj.rejectGround : true,
    // Opt-in only: anything other than a literal `true` (missing, legacy client, bad type) keeps the classical ridge detector off.
    enableClassicalRidge: obj.enableClassicalRidge === true,
    useSiteReferenceForNonGps:
      typeof obj.useSiteReferenceForNonGps === 'boolean'
        ? obj.useSiteReferenceForNonGps
        : false, // no invented location unless explicitly enabled
    siteReferenceLat:
      typeof obj.siteReferenceLat === 'number' && Number.isFinite(obj.siteReferenceLat)
        ? obj.siteReferenceLat
        : 40.65896423,
    siteReferenceLon:
      typeof obj.siteReferenceLon === 'number' && Number.isFinite(obj.siteReferenceLon)
        ? obj.siteReferenceLon
        : -104.86752766,
  };
}

function isSupportedImageEntry(name: string): boolean {
  const lower = name.toLowerCase();
  if (lower.includes('__macosx/') || path.basename(lower).startsWith('.')) return false;
  return /\.(jpe?g|jfif|png|webp|tiff?)$/i.test(lower);
}

function mergeServerGeoJson(
  results: InspectionResult[],
  onlyImagesWithIssues: boolean
): GeoJsonFeatureCollection {
  const features: GeoJsonPointFeature[] = [];
  let skippedNoGeo = 0;
  for (const r of results) {
    const imgFeature = r.geoJson.features[0];
    if (!imgFeature) continue;
    if (imgFeature.properties.crs_mode === 'PIXEL_LOCAL') {
      skippedNoGeo++; // pixel coords don't belong in an EPSG:3857 layer
      continue;
    }
    if (onlyImagesWithIssues && imgFeature.properties.total_anomaly_count === 0) continue;
    features.push(imgFeature);
  }

  const imagesWithSpiderCount = features.filter(
    (f) => f.properties.has_spider_web_crack
  ).length;
  const totalSpiderCount = features.reduce(
    (acc, f) => acc + f.properties.spider_web_crack_count,
    0
  );
  const totalEdgeSpiderCount = features.reduce(
    (acc, f) => acc + f.properties.edge_spider_web_crack_count,
    0
  );

  return {
    type: 'FeatureCollection',
    name: 'pv_batch_image_crack_inspection',
    crs: {
      type: 'name',
      properties: {
        name: 'urn:ogc:def:crs:EPSG::3857',
      },
    },
    metadata: {
      crs_epsg: 'EPSG:3857',
      crs_label: 'EPSG:3857 - WGS 84 / Pseudo-Mercator',
      generated_at: new Date().toISOString(),
      total_images: results.length,
      total_image_points: features.length,
      images_with_spider_web_cracks: imagesWithSpiderCount,
      total_spider_web_cracks: totalSpiderCount,
      total_edge_spider_web_cracks: totalEdgeSpiderCount,
      skipped_without_georeference: skippedNoGeo,
      coordinate_system_note:
        'One GeoJSON Point [x_easting_m, y_northing_m] per image in EPSG:3857 - WGS 84 / Pseudo-Mercator (from GeoTIFF / DJI RTK XMP / EXIF GPS or configured Site Reference), with all detected crack details in properties.',
    },
    features,
  };
}

// Desktop mode binds loopback only. Remote mode is an explicit opt-in and requires a strong shared token.
const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '::1']);
const REQUESTED_HOST = (process.env.HOST || '127.0.0.1').trim();
const REMOTE_TOKEN = process.env.PV_REMOTE_TOKEN?.trim() ?? '';
const REMOTE_ENABLED = process.env.PV_ALLOW_REMOTE_BIND === '1' && REMOTE_TOKEN.length >= 32;
const BIND_HOST = LOOPBACK_HOSTS.has(REQUESTED_HOST) || REMOTE_ENABLED ? REQUESTED_HOST : '127.0.0.1';
if (BIND_HOST !== REQUESTED_HOST) console.warn(`HOST=${REQUESTED_HOST} ignored: binding to 127.0.0.1. Remote access needs PV_ALLOW_REMOTE_BIND=1 and a PV_REMOTE_TOKEN of 32+ characters.`);

async function startServer() {
  const app = express();
  app.disable('x-powered-by');
  app.disable('etag');
  const PORT = Number(process.env.PORT) || 3000; // hosts inject PORT
  const isProduction = process.env.NODE_ENV === 'production';
  const csp = [
    "default-src 'self'",
    // Dev only: Vite's React plugin injects an inline Fast-Refresh preamble; without 'unsafe-inline' React never mounts (blank page).
    // Production serves only the built same-origin bundle, so it keeps the strict policy.
    `script-src 'self'${isProduction ? '' : " 'unsafe-inline' 'unsafe-eval'"}`,
    "style-src 'self' 'unsafe-inline'",
    "font-src 'self' data:",
    "img-src 'self' data: blob:",
    `connect-src 'self'${isProduction ? '' : ' ws: wss:'}`,
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
  ].join('; ');
  app.use((req: Request, res: Response, next: NextFunction) => {
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('Referrer-Policy', 'no-referrer');
    res.setHeader('X-DNS-Prefetch-Control', 'off');
    res.setHeader('X-Permitted-Cross-Domain-Policies', 'none');
    res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
    res.setHeader('Content-Security-Policy', csp);
    if (req.path.startsWith('/api/')) res.setHeader('Cache-Control', 'no-store');
    const decision = validateRequestSecurity({
      hostHeader: req.get('host'),
      originHeader: req.get('origin'),
      fetchSite: req.get('sec-fetch-site'),
      remoteAddress: req.socket.remoteAddress,
      port: PORT,
      remoteEnabled: REMOTE_ENABLED,
      remoteToken: REMOTE_TOKEN,
      suppliedToken: req.get('x-pv-token'),
    });
    if (!decision.allowed) {
      res.status(decision.reason === 'cross-site' || decision.reason === 'invalid-origin' ? 403 : 421)
        .json({ error: 'Request rejected by the local application security policy.' });
      return;
    }
    next();
  });
  app.use('/api', rateLimit);
  app.use('/api/inspect/upload', express.raw({ type: '*/*', limit: '100mb' })); // v1.4.11: equals the cell-detect limit so an image cannot pass stage 1 and fail stage 2
  app.use('/api/inspect/upload-zip', express.raw({ type: '*/*', limit: '200mb' }));
  app.use(express.json({ limit: '20mb' }));

  app.get('/api/health', (_req, res) => {
    res.json({
      ok: true,
      engine: 'classical-offline',
      models: getModelHealth(),
      timestamp: new Date().toISOString(),
    });
  });

  // Load raw-image metadata (camera GPS per filename) used when an image has no embedded GPS
  app.get('/api/metadata', (_req, res) => {
    res.json({ records: metadataCount() });
  });
  app.post('/api/metadata', (req, res) => {
    try {
      const n = loadMetadata(req.body);
      res.json({ records: n });
    } catch (err: unknown) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Invalid metadata JSON.' });
    }
  });

  // ---- Direct PV-cell inference controls ----
  app.get('/api/cell-detector', (_req, res) => res.json(getCellDetectorStatus()));
  app.post('/api/cell-detector', (req, res) => { try { setCellDetectorMode(req.body?.mode); res.json(getCellDetectorStatus()); } catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Invalid cell detector mode.' }); } }); // bad mode must not 500/crash

  // First-stage, image-only YOLO pass. This route deliberately does not invoke crack analysis.
  app.post('/api/cell-detect/upload', express.raw({ type: '*/*', limit: '100mb' }), async (req, res) => {
    try {
      if (!Buffer.isBuffer(req.body) || req.body.length === 0) { res.status(400).json({ error: 'Empty image payload uploaded.' }); return; }
      const rawName = req.headers['x-file-name'];
      const fileName = typeof rawName === 'string' && rawName.trim() ? safeDecode(rawName) : `Cell_Detection_${Date.now()}.jpg`;
      const maxNativeDim = 8000;
      const metadata = await sharp(req.body).rotate().metadata();
      const originalWidth = metadata.width || 0, originalHeight = metadata.height || 0;
      if (!originalWidth || !originalHeight) { res.status(400).json({ error: 'Could not read source image dimensions.' }); return; }
      const nativeScale = Math.min(1, maxNativeDim / Math.max(originalWidth, originalHeight));
      let imagePipeline = sharp(req.body).rotate();
      if (nativeScale < 1) imagePipeline = imagePipeline.resize(Math.max(32, Math.round(originalWidth * nativeScale)), Math.max(32, Math.round(originalHeight * nativeScale)), { fit: 'fill' });
      const decoded = await imagePipeline.toColorspace('srgb').removeAlpha().raw({ depth: 'uchar' }).toBuffer({ resolveWithObject: true });
      let rgb = decoded.data;
      if (decoded.info.channels === 1) {
        const expanded = Buffer.alloc(decoded.info.width * decoded.info.height * 3);
        for (let index = 0; index < decoded.info.width * decoded.info.height; index++) expanded.fill(decoded.data[index], index * 3, index * 3 + 3);
        rgb = expanded;
      } else if (decoded.info.channels !== 3) {
        throw new Error(`Unsupported normalized image channel count: ${decoded.info.channels}`);
      }
      const detection = await detectCellBoxes(rgb, decoded.info.width, decoded.info.height, req.headers['x-debug-inputs'] === '1');
      const boxes = detection.boxes.map(({ left, top, width, height }) => ({ left, top, width, height }));
      res.setHeader('Cache-Control', 'no-store');
      res.json({
        imageId: typeof req.headers['x-frame-id'] === 'string' ? req.headers['x-frame-id'] : `${fileName}-${Date.now()}`,
        fileName, imageWidth: decoded.info.width, imageHeight: decoded.info.height,
        originalWidth, originalHeight, boxes, source: detection.source,
        diagnostics: detection.diagnostics, warning: detection.warning,
        ...(detection.inputPreviews ? { inputPreviews: detection.inputPreviews } : {}),
      });
    } catch (error) {
      res.status(500).json({ error: error instanceof Error ? error.message : 'Cell detection failed.' });
    }
  });

  // ---- Direct cell-box annotation for YOLO cell training ----
  app.post('/api/cell-annotations/negative', express.raw({ type: '*/*', limit: '100mb' }), async (req, res) => {
    try {
      if (!Buffer.isBuffer(req.body) || req.body.length === 0) { res.status(400).json({ error: 'Empty negative-sample image payload.' }); return; }
      const separator = req.body.indexOf(10);
      if (separator <= 0) { res.status(400).json({ error: 'Negative-sample metadata header is missing.' }); return; }
      const meta = JSON.parse(req.body.subarray(0, separator).toString('utf8')) as { fileName?: string };
      const image = req.body.subarray(separator + 1);
      if (!image.length) { res.status(400).json({ error: 'Negative-sample image is empty.' }); return; }
      const normalized = await sharp(image).rotate().jpeg().toBuffer({ resolveWithObject: true });
      if (!normalized.info.width || !normalized.info.height) { res.status(400).json({ error: 'Could not read source image dimensions.' }); return; }
      const saved = saveNegativeCellSample(normalized.data, typeof meta.fileName === 'string' ? path.basename(meta.fileName) : 'negative-background-image');
      const records = listCellAnnotations();
      res.json({ saved: true, sampleType: 'negative', id: saved.id, negativeSamples: records.filter((record) => record.sampleType === 'negative' && !record.boxes.length).length });
    } catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not save the negative sample.' }); }
  });
  app.post('/api/cell-annotations', express.raw({ type: '*/*', limit: '100mb' }), async (req, res) => {
    try {
      if (!Buffer.isBuffer(req.body) || req.body.length === 0) { res.status(400).json({ error: 'Empty cell annotation image payload.' }); return; }
      const separator = req.body.indexOf(10);
      if (separator <= 0) { res.status(400).json({ error: 'Cell annotation metadata header is missing.' }); return; }
      const meta = JSON.parse(req.body.subarray(0, separator).toString('utf8')) as { fileName?: string; imageId?: string; partial?: boolean; coordinateWidth?: number; coordinateHeight?: number; missedBoxes?: Array<{ left: number; top: number; width: number; height: number }>; boxes?: Array<{ left: number; top: number; width: number; height: number }> };
      const image = req.body.subarray(separator + 1);
      if (!image.length || !Array.isArray(meta.boxes) || meta.boxes.length < 1 || meta.boxes.length > 3000) { res.status(400).json({ error: 'Draw and save between 1 and 3,000 individual cell boxes.' }); return; }
      const normalized = await sharp(image).rotate().jpeg().toBuffer({ resolveWithObject: true });
      const imageWidth = normalized.info.width, imageHeight = normalized.info.height;
      if (!imageWidth || !imageHeight) { res.status(400).json({ error: 'Could not read source image dimensions.' }); return; }
      const boxes = scaleCellBoxesToSourceImage(meta.boxes, imageWidth, imageHeight, meta.coordinateWidth, meta.coordinateHeight);
      // v1.4.11.5: operator-added cells (missed by the model) are scaled like the boxes and stored for hard-example oversampling.
      const missedBoxes = Array.isArray(meta.missedBoxes) && meta.missedBoxes.length ? scaleCellBoxesToSourceImage(meta.missedBoxes.slice(0, 3000), imageWidth, imageHeight, meta.coordinateWidth, meta.coordinateHeight) : [];
      // Persist the exact inference source image (not an older training-set image)
      // with coordinates converted to its full, EXIF-normalized pixel space.
      const saved = saveCellAnnotations(normalized.data, typeof meta.fileName === 'string' ? path.basename(meta.fileName) : 'image', boxes, typeof meta.imageId === 'string' ? meta.imageId : undefined, missedBoxes, typeof meta.partial === 'boolean' ? meta.partial : undefined);
      const annotationRecords = listCellAnnotations();
      res.json({ saved: true, missedCells: saved.missedBoxes?.length ?? 0, id: saved.id, sourceImageId: saved.sourceImageId, imageFile: saved.imageFile, fileName: saved.fileName, boxes: saved.boxes.length, images: annotationRecords.length, positiveImages: annotationRecords.filter((record) => record.boxes.length > 0).length });
    } catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not save cell annotations.' }); }
  });
  // v1.4.12.6: persistent per-image review state (keyed by the SHA-1 of the image bytes, computed in the browser).
  app.get('/api/review/:hash', (req, res) => {
    res.setHeader('Cache-Control', 'no-store');
    if (!isValidReviewHash(req.params.hash)) { res.status(400).json({ error: 'Invalid image hash.' }); return; }
    const record = loadReview(req.params.hash);
    if (!record) { res.status(404).json({ error: 'No saved review for this image.' }); return; }
    res.json(record);
  });
  app.put('/api/review/:hash', (req, res) => {
    try { const record = saveReview(req.params.hash, req.body); res.json({ saved: true, savedAt: record.savedAt }); }
    catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not save review.' }); }
  });
  app.delete('/api/review/:hash', (req, res) => {
    try { res.json({ deleted: deleteReview(req.params.hash) }); }
    catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not delete review.' }); }
  });
  app.get('/api/cell-annotations', (_req, res) => { const records = listCellAnnotations(); res.json({ images: records.filter((record) => (record.boxes?.length ?? 0) > 0).length, negativeSamples: records.filter((record) => record.sampleType === 'negative' && !(record.boxes?.length)).length, boxes: records.reduce((sum, record) => sum + record.boxes.length, 0) }); });

  // v1.4.7.0: measure the cell type (aspect ratio) from the operator's saved cell boxes and make it the default accept profile.
  app.post('/api/cell-type-filter/learn', (_req, res) => {
    try {
      const boxes = listCellAnnotations().flatMap((record) => record.boxes ?? []);
      const learned = learnProfileFromBoxes(boxes);
      if (!learned) { res.status(400).json({ error: `Need at least 4 saved cell boxes to learn a cell type; found ${boxes.length}. Save reviewed boxes in Cell Annotation first.` }); return; }
      const current = getTrainingSettings();
      const profiles = current.cellTypeFilter.profiles.filter((profile) => profile.id !== 'learned');
      profiles.unshift({ id: 'learned', name: `Learned from annotations (${learned.samples} boxes)`, mode: 'accept', enabled: true, aspect: learned.aspect, aspectTolerance: learned.aspectTolerance, minLongPx: 0, maxLongPx: 0 });
      // The learned type replaces the generic default; extra types the operator added stay.
      const next = profiles.map((profile) => profile.id === 'standard' ? { ...profile, enabled: false } : profile);
      res.json({ learned, settings: saveTrainingSettings({ ...current, cellTypeFilter: { ...current.cellTypeFilter, profiles: next } }) });
    } catch (error) { res.status(500).json({ error: error instanceof Error ? error.message : 'Could not learn the cell type.' }); }
  });

  // ---- Shared training hyperparameters for both model families ----
  app.get('/api/training-settings', (_req, res) => { res.setHeader('Cache-Control', 'no-store'); res.json(getTrainingSettings()); });
  app.put('/api/training-settings', (req, res) => { try { res.json(saveTrainingSettings(req.body)); } catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not save training settings.' }); } });

  // ---- v1.4.8.0: persistent data folder, model import/export, settings reset ----
  // These endpoints touch the file system, so they accept only loopback clients and require a custom header
  // (a custom header forces a CORS pre-flight, which this server never grants to other web pages).
  const localOnly = (req: Request, res: Response, next: NextFunction) => {
    const loopback = isLoopbackAddress(req.socket.remoteAddress ?? '');
    if (!loopback || req.get('x-pv-action') !== '1') { res.status(403).json({ error: 'This action is only available from the local PV CrackVision window.' }); return; }
    next();
  };
  const appVersion = APP_VERSION.replace(/-(\d+)$/, '.$1'); // single source: package.json (see server/appVersion.ts)
  app.get('/api/app/info', (_req, res) => { res.setHeader('Cache-Control', 'no-store'); res.json({ version: appVersion, packaged: isPackagedApp(), dataFolder: describeDataFolder() }); });
  app.get('/api/data-folder/usage', localOnly, (_req, res) => { dataFolderUsage().then((usage) => res.json(usage)).catch((error) => res.status(500).json({ error: error instanceof Error ? error.message : 'Could not measure the data folder.' })); });
  app.get('/api/data-folder/browse', localOnly, (req, res) => res.json(browseFolders(typeof req.query.path === 'string' ? req.query.path : undefined)));
  app.post('/api/data-folder', localOnly, async (req, res) => {
    try {
      const result = await setDataFolder(String(req.body?.path ?? ''), { copyExisting: req.body?.copyExisting !== false, includeRuntime: req.body?.includeRuntime === true });
      res.json({ ...result, info: describeDataFolder() });
    } catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not set the data folder.' }); }
  });
  app.post('/api/data-folder/reset', localOnly, (_req, res) => { try { res.json({ ...resetDataFolderToDefault(), info: describeDataFolder() }); } catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not reset the data folder.' }); } });
  // v1.4.11: the packaged app has no supervisor, so the Electron shell registers a relaunch hook (app.relaunch + exit).
  // Source mode keeps exit code 75, which scripts/run-server.mjs turns into a restart with the new data folder.
  app.post('/api/app/restart', localOnly, (_req, res) => {
    res.json({ restarting: true });
    const desktopRestart = (globalThis as { __pvDesktopRestart?: () => void }).__pvDesktopRestart;
    setTimeout(() => { if (typeof desktopRestart === 'function') desktopRestart(); else process.exit(75); }, 400);
  });
  // Portable, merge-only backup/restore for direct cell-detector labels and source images.
  // This is separate from crack-classifier feedback and never changes the active detector model.
  app.get('/api/cell-training/data/export', localOnly, async (_req, res) => {
    try {
      const records = listCellAnnotations();
      const archive = await createCellTrainingDataBundle(records, (record) => fs.promises.readFile(cellAnnotationImagePath(record)), APP_VERSION, getCellTrainingCheckpointBuffer);
      res.setHeader('Cache-Control', 'no-store');
      res.setHeader('Content-Type', 'application/zip');
      res.setHeader('Content-Disposition', 'attachment; filename="pv-crackvision-cell-training-data.zip"');
      res.send(archive);
    } catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not export cell training data.' }); }
  });
  app.post('/api/cell-training/data/import', localOnly, express.raw({ type: '*/*', limit: '600mb' }), async (req, res) => {
    try {
      const current = getCellTrainingStatus();
      if (['setting_up', 'training', 'paused'].includes(current.status)) { res.status(409).json({ error: 'Finish or abort cell training before importing a data backup.' }); return; }
      if (!Buffer.isBuffer(req.body) || req.body.length === 0) { res.status(400).json({ error: 'The selected cell training backup is empty.' }); return; }
      const bundle = await parseCellTrainingDataBundle(req.body);
      for (const [imageFile, bytes] of bundle.images) {
        const record = bundle.records.find((item) => item.imageFile === imageFile);
        if (!record) continue;
        const destination = cellAnnotationImagePath(record);
        if (!(await fs.promises.access(destination).then(() => true, () => false))) {
          await fs.promises.mkdir(path.dirname(destination), { recursive: true });
          const temporary = `${destination}.${process.pid}.import.tmp`;
          await fs.promises.writeFile(temporary, bytes);
          try { await fs.promises.rename(temporary, destination); }
          finally { await fs.promises.rm(temporary, { force: true }); }
        }
      }
      const counts = mergeImportedCellAnnotations(bundle.records);
      let checkpointRestored = false;
      if (bundle.checkpoint) {
        const checkpointPath = path.join(DATA_ROOT, 'training', 'models', 'cell-best.pt');
        if (!(await fs.promises.access(checkpointPath).then(() => true, () => false))) {
          await fs.promises.mkdir(path.dirname(checkpointPath), { recursive: true });
          const temporary = `${checkpointPath}.${process.pid}.import.tmp`;
          await fs.promises.writeFile(temporary, bundle.checkpoint);
          try { await fs.promises.rename(temporary, checkpointPath); checkpointRestored = true; }
          finally { await fs.promises.rm(temporary, { force: true }); }
        }
      }
      res.json({ ...getCellTrainingStatus(), importedAnnotations: counts.imported, existingAnnotations: counts.alreadyPresent, checkpointRestored, backupAppVersion: bundle.manifest.appVersion, backupCreatedAt: bundle.manifest.createdAt });
    } catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not import the cell training backup.' }); }
  });
  app.post('/api/cell-training/reset', localOnly, (_req, res) => {
    try { res.json(resetCellTrainingDataAndModel()); }
    catch (error) { res.status(409).json({ error: error instanceof Error ? error.message : 'Could not reset cell training data and model.' }); }
  });

  app.post('/api/cell-model/import', localOnly, express.raw({ type: '*/*', limit: '600mb' }), async (req, res) => {
    try {
      const name = decodeURIComponent(String(req.get('x-file-name') ?? 'model'));
      res.json(await importModelFile(Buffer.isBuffer(req.body) ? req.body : Buffer.alloc(0), name));
    } catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not load the model.' }); }
  });
  app.get('/api/cell-model/export', localOnly, (_req, res) => {
    const active = resolveCellModel().candidates[0];
    if (!active) { res.status(404).json({ error: 'There is no active cell model to export yet.' }); return; }
    res.download(active.path, 'cell.onnx');
  });
  app.post('/api/training-settings/reset', localOnly, (_req, res) => { try { res.json(saveTrainingSettings(DEFAULT_TRAINING_SETTINGS)); } catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not reset training settings.' }); } });

  // ---- In-app YOLO11 direct cell training ----
  app.get('/api/cell-training/status', (_req, res) => { res.setHeader('Cache-Control', 'no-store'); res.json(getCellTrainingStatus()); });
  app.get('/api/cell-training/metrics.csv', (_req, res) => {
    const status = getCellTrainingStatus();
    if (!(status.epochMetrics?.length)) { res.status(404).type('text/plain').send('No epoch metrics have been recorded yet.'); return; }
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('Content-Disposition', 'attachment; filename="pv-crackvision-cell-epoch-metrics.csv"');
    res.type('text/csv').send(getCellTrainingMetricsCsv());
  });
  app.get('/api/cell-training/log', (_req, res) => {
    const logFile = getCellTrainingLogPath();
    if (!logFile) { res.status(404).type('text/plain').send('No training log is available yet.'); return; }
    res.setHeader('Cache-Control', 'no-store');
    res.download(logFile, path.basename(logFile));
  });
  app.post('/api/cell-training/setup', (_req, res) => {
    if (['setting_up', 'training', 'paused'].includes(getCellTrainingStatus().status)) { res.status(409).json({ error: 'A cell training operation is already running.' }); return; }
    void setupCellTrainingEngine().catch(() => undefined); res.json(getCellTrainingStatus());
  });
  app.post('/api/cell-training/start', async (req, res) => {
    try { if (req.body && typeof req.body === 'object') saveTrainingSettings({ ...getTrainingSettings(), cellDetector: { ...getTrainingSettings().cellDetector, ...req.body } }); res.json(await startCellYoloTraining(req.body && typeof req.body === 'object' ? req.body : {})); }
    catch (error) { res.status(400).json({ ...getCellTrainingStatus(), error: error instanceof Error ? error.message : 'Could not start cell training.' }); }
  });
  app.post('/api/cell-training/pause', (_req, res) => { try { res.json(pauseCellYoloTraining()); } catch (error) { res.status(500).json({ error: error instanceof Error ? error.message : 'Pause failed.' }); } });
  app.post('/api/cell-training/resume', (_req, res) => { try { res.json(resumeCellYoloTraining()); } catch (error) { res.status(500).json({ error: error instanceof Error ? error.message : 'Resume failed.' }); } });
  app.post('/api/cell-training/cancel', (_req, res) => { try { res.json(cancelCellYoloTraining()); } catch (error) { res.status(500).json({ error: error instanceof Error ? error.message : 'Cancel failed.' }); } });
  app.post('/api/cell-training/rollback', (_req, res) => { try { res.json(rollbackCellModel()); } catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not roll back the cell model.' }); } });

  // ---- Reviewed cell annotations + fresh cell-level model training ----
  const feedbackState = () => ({ ...feedbackCounts() });
  app.get('/api/feedback', (_req, res) => {
    try { res.json({ ...feedbackState(), items: listFeedback(), lastUndo: getLastFeedbackUndo() }); } catch (err) { res.status(500).json({ error: err instanceof Error ? err.message : 'Could not read feedback.' }); }
  });
  app.post('/api/feedback/batch', express.raw({ type: '*/*', limit: '100mb' }), async (req, res) => {
    try {
      if (!Buffer.isBuffer(req.body) || req.body.length === 0) {
        res.status(400).json({ error: 'Empty image payload.' });
        return;
      }
      const separator = req.body.indexOf(10);
      if (separator <= 0) { res.status(400).json({ error: 'Batch metadata header is missing.' }); return; }
      if (separator > 1024 * 1024) { res.status(413).json({ error: 'Batch metadata header exceeds the 1 MB safety limit.' }); return; }
      const metadata = JSON.parse(req.body.subarray(0, separator).toString('utf8')) as {
        fileName?: string;
        coordinateWidth?: number;
        coordinateHeight?: number;
        replace?: boolean;
        transactional?: boolean;
        annotations?: Array<{ left: number; top: number; width: number; height: number; label: 0 | 1; category?: string; source?: string; modelScore?: number; moduleBox?: { left: number; top: number; width: number; height: number } }>;
      };
      const imageBytes = req.body.subarray(separator + 1);
      const annotations = metadata.annotations;
      if (!imageBytes.length || !Array.isArray(annotations) || annotations.length < 1 || annotations.length > 5000) {
        res.status(400).json({ error: 'Provide image data and between 1 and 5,000 cell annotations.' });
        return;
      }
      const rawFileName = typeof metadata.fileName === 'string' ? metadata.fileName : 'image';
      const fileName = rawFileName.replace(/\\/g, '/').split('/').pop()?.replace(/[\u0000-\u001f\u007f]/g, '').slice(0, 260) || 'image';
      // Coordinates are supplied in the current preview frame and stored in the EXIF-oriented source image space.
      const dims = await sharp(imageBytes, { failOn: 'error' }).rotate().metadata();
      // v1.4.17.1: metadata() after .rotate() still reports the STORED size; orientations 5-8 swap width/height (same rule as cvPipeline).
      const swapXY = (dims.orientation ?? 1) >= 5;
      const srcW = (swapXY ? dims.height : dims.width) || 0, srcH = (swapXY ? dims.width : dims.height) || 0;
      if (!srcW || !srcH || !dims.format || !['jpeg', 'png', 'webp', 'tiff', 'gif', 'heif', 'avif'].includes(dims.format)) {
        res.status(400).json({ error: 'The uploaded source bytes are not a supported raster image.' }); return;
      }
      const widthProvided = metadata.coordinateWidth !== undefined;
      const heightProvided = metadata.coordinateHeight !== undefined;
      if (widthProvided !== heightProvided) { res.status(400).json({ error: 'Coordinate width and height must be provided together.' }); return; }
      const hasSpace = widthProvided && heightProvided;
      if (hasSpace && (!Number.isFinite(metadata.coordinateWidth) || !Number.isFinite(metadata.coordinateHeight) || Number(metadata.coordinateWidth) <= 0 || Number(metadata.coordinateHeight) <= 0 || Number(metadata.coordinateWidth) > 1_000_000 || Number(metadata.coordinateHeight) > 1_000_000)) {
        res.status(400).json({ error: 'Coordinate dimensions are outside the supported range.' }); return;
      }
      const coordW = hasSpace ? Number(metadata.coordinateWidth) : srcW;
      const coordH = hasSpace ? Number(metadata.coordinateHeight) : srcH;
      const kx = srcW / coordW;
      const ky = srcH / coordH;
      const metas = annotations.map((entry, index) => {
        const values = [entry.left, entry.top, entry.width, entry.height];
        if (values.some((n) => typeof n !== 'number' || !Number.isFinite(n)) || entry.left < 0 || entry.top < 0 || entry.width <= 0 || entry.height <= 0 || entry.left + entry.width > coordW + 1 || entry.top + entry.height > coordH + 1 || (entry.label !== 0 && entry.label !== 1)) {
          throw new Error(`Annotation ${index + 1} has invalid coordinates or label, or extends outside its source image.`);
        }
        if (entry.category !== undefined && entry.category !== 'spider_web_crack') throw new Error(`Annotation ${index + 1} has an unsupported feedback category.`);
        if (entry.source !== undefined) {
          const validSource = entry.source === 'rejected-detection' || entry.source === 'missed-crack' || entry.source === 'bulk-negative' || entry.source === 'auto-negative';
          const sourceMatchesLabel = (entry.source === 'rejected-detection' || entry.source === 'bulk-negative' || entry.source === 'auto-negative') ? entry.label === 0 : entry.source === 'missed-crack' ? entry.label === 1 : false;
          if (!validSource || !sourceMatchesLabel) throw new Error(`Annotation ${index + 1} has an invalid hard-negative / missed-crack source tag for its label.`);
        }
        if (entry.modelScore !== undefined && (typeof entry.modelScore !== 'number' || !Number.isFinite(entry.modelScore) || entry.modelScore < 0 || entry.modelScore > 1)) {
          throw new Error(`Annotation ${index + 1} has an invalid model score; expected a number from 0 to 1.`);
        }
        const mb = entry.moduleBox;
        if (mb && [mb.left, mb.top, mb.width, mb.height].some((n) => typeof n !== 'number' || !Number.isFinite(n)) || mb && (mb.left < 0 || mb.top < 0 || mb.width <= 0 || mb.height <= 0 || mb.left + mb.width > coordW + 1 || mb.top + mb.height > coordH + 1)) {
          throw new Error(`Annotation ${index + 1} has an invalid or out-of-bounds module outline.`);
        }
        const left = Math.round(entry.left * kx), top = Math.round(entry.top * ky);
        const width = Math.max(1, Math.round(entry.width * kx)), height = Math.max(1, Math.round(entry.height * ky));
        if (left + width > srcW + 1 || top + height > srcH + 1) throw new Error(`Annotation ${index + 1} is outside the decoded source image.`);
        return {
          fileName,
          left,
          top,
          width,
          height,
          label: entry.label,
          category: 'spider_web_crack',
          ...(entry.source ? { source: entry.source as 'rejected-detection' | 'missed-crack' | 'bulk-negative' | 'auto-negative' } : {}),
          ...(entry.modelScore !== undefined ? { modelScore: entry.modelScore } : {}),
          ...(mb ? { moduleLeft: Math.round(mb.left * kx), moduleTop: Math.round(mb.top * ky), moduleWidth: Math.max(1, Math.round(mb.width * kx)), moduleHeight: Math.max(1, Math.round(mb.height * ky)) } : {}),
        };
      });
      const result = metadata.transactional === true
        ? addFeedbackManyTransactional(imageBytes, metas, { replace: metadata.replace === true })
        : addFeedbackMany(imageBytes, metas, { replace: metadata.replace === true });
      const items = result.saved.map((saved, index) => ({ index, id: saved.id }));
      res.json({ items, replaced: result.replaced, ...('undoToken' in result ? { undoToken: result.undoToken } : {}), ...feedbackState() });
    } catch (err: unknown) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Could not save cell feedback.' });
    }
  });
  app.get('/api/feedback/existing/:imageFile', (req, res) => {
    if (!/^[0-9a-f]{20}\.img$/.test(req.params.imageFile)) { res.status(400).json({ error: 'Invalid image id.' }); return; }
    res.json(existingFeedbackForImage(req.params.imageFile));
  });
  app.delete('/api/feedback/:id', (req, res) => {
    try { res.json({ removed: removeFeedback(req.params.id), ...feedbackState() }); } catch (err) { res.status(500).json({ error: err instanceof Error ? err.message : 'Could not remove feedback.' }); } // fs errors
  });
  app.post('/api/feedback/undo-batch', express.json({ limit: '16kb' }), (req, res) => {
    try {
      const token = typeof req.body?.undoToken === 'string' ? req.body.undoToken : '';
      if (!token) { res.status(400).json({ error: 'A feedback undo token is required.' }); return; }
      const restored = undoFeedbackBatch(token);
      if (!restored) { res.status(409).json({ error: 'This feedback batch can no longer be undone (it may have expired or a newer edit changed one of its cells).' }); return; }
      res.json({ restored: true, ...feedbackState(), lastUndo: getLastFeedbackUndo() });
    } catch (err) { res.status(500).json({ error: err instanceof Error ? err.message : 'Could not undo the feedback batch.' }); }
  });
  // Crack models: 'anom' (anomaly detector, default engine); archived 'cls' (classifier A) and 'det' (shadow detector B). They train independently and can run at the same time.
  app.get('/api/classifier/status', (_req, res) => {
    res.setHeader('Cache-Control', 'no-store');
    res.json(classifierStatusSnapshot());
  });
  // v1.4.14.4: remove the installed A/B models and their generated training data (reviewed labels are kept). 409 while a job is running.
  // scope: archived (A + B, default) or anomaly (the anomaly detector).
  // v1.4.17.7: models from earlier releases + the crack training-data tab
  app.get('/api/models/recovery', (_req, res) => { res.setHeader('Cache-Control', 'no-store'); res.json(getLastRecoveryReport()); });
  app.post('/api/models/recover', localOnly, async (_req, res) => {
    try { const report = recoverPreviousModels(); for (const engine of ['anomaly', 'classifier'] as const) await loadCellClassifier(engine).catch(() => null); res.json({ report, status: classifierStatusSnapshot() }); }
    catch (error) { res.status(500).json({ error: error instanceof Error ? error.message : 'Model recovery failed.' }); }
  });
  app.get('/api/crack-training/summary', (_req, res) => {
    try {
      const items = listFeedback().filter((i) => i.category === 'spider_web_crack');
      const bySource: Record<string, { cracked: number; clean: number }> = {}, images = new Map<string, { imageFile: string; fileName: string; cracked: number; clean: number; lastAt: string }>();
      for (const i of items) {
        const key = i.source ?? 'reviewed', s = (bySource[key] ??= { cracked: 0, clean: 0 }); i.label === 1 ? s.cracked++ : s.clean++;
        const img = images.get(i.imageFile) ?? { imageFile: i.imageFile, fileName: i.fileName, cracked: 0, clean: 0, lastAt: '' }; i.label === 1 ? img.cracked++ : img.clean++; if (i.createdAt > img.lastAt) img.lastAt = i.createdAt; images.set(i.imageFile, img);
      }
      res.setHeader('Cache-Control', 'no-store');
      res.json({ counts: feedbackCounts(), bySource, images: [...images.values()].sort((a, b) => b.lastAt.localeCompare(a.lastAt)), items: items.map(({ id, fileName, imageFile, label, source, modelScore, createdAt, width, height }) => ({ id, fileName, imageFile, label, source: source ?? 'reviewed', modelScore, createdAt, width, height })) });
    } catch (error) { res.status(500).json({ error: error instanceof Error ? error.message : 'Could not read the crack training data.' }); }
  });
  // One training cell as a small JPEG (cropped from the stored source image with the same buffer the models use).
  app.get('/api/crack-training/cell/:id', async (req, res) => {
    try {
      if (!/^[0-9a-f-]{36}$/i.test(req.params.id)) { res.status(400).json({ error: 'Invalid id.' }); return; }
      const item = listFeedback().find((i) => i.id === req.params.id);
      if (!item) { res.status(404).json({ error: 'Not found.' }); return; }
      const img = sharp(feedbackImagePath(item)).rotate(), meta = await img.metadata();
      const box = bufferedCellCrop(meta.width ?? 0, meta.height ?? 0, item, 0.03);
      if (box.width < 2 || box.height < 2) { res.status(422).json({ error: 'Cell is outside the image.' }); return; }
      const jpeg = await sharp(feedbackImagePath(item)).rotate().extract(box).resize(160, 160, { fit: 'inside' }).jpeg({ quality: 82 }).toBuffer();
      res.setHeader('Content-Type', 'image/jpeg'); res.setHeader('Cache-Control', 'private, max-age=300'); res.send(jpeg);
    } catch (error) { res.status(500).json({ error: error instanceof Error ? error.message : 'Could not render the cell.' }); }
  });
  app.post('/api/classifier/clear', localOnly, (req, res) => {
    try { res.json(clearClassifierModelsAndData(req.query.scope === 'anomaly' ? 'anomaly' : 'archived')); } catch (error) { res.status(409).json({ error: error instanceof Error ? error.message : 'Could not clear the crack models.' }); }
  });
  app.post('/api/classifier/:mode/start', (req, res) => {
    if (!isClassifierMode(req.params.mode)) { res.status(400).json({ error: 'Unknown model (use anom, cls or det).' }); return; }
    try { res.json(startClassifierTraining(req.params.mode)); } catch (error) { res.status(409).json({ error: error instanceof Error ? error.message : 'Could not start training.' }); }
  });
  app.post('/api/classifier/:mode/cancel', (req, res) => {
    if (!isClassifierMode(req.params.mode)) { res.status(400).json({ error: 'Unknown model (use anom, cls or det).' }); return; }
    res.json(cancelClassifierTraining(req.params.mode));
  });
  app.get('/api/classifier/:mode/log', (req, res) => {
    if (!isClassifierMode(req.params.mode)) { res.status(400).json({ error: 'Unknown model (use anom, cls or det).' }); return; }
    res.type('text/plain').attachment(`crack-${req.params.mode}-training.log`).send(classifierLogText(req.params.mode));
  });
  app.delete('/api/train/reset', (_req, res) => {
    try { clearFeedbackDataset(); res.json(feedbackState()); } catch (err) { res.status(500).json({ error: err instanceof Error ? err.message : 'Reset failed.' }); } // fs errors (e.g. Windows locked files)
  });

  // Analyze single image frame (used for individual uploads, folder files, and client-streamed zip entries).
  // v1.4.9: the app sends the cell boxes from the Cell Annotation stage with the image so the crack pass inspects exactly
  // those cells instead of re-deriving a classical grid. Wire format (header `x-inspect-envelope: 1`): one JSON line
  //   {"cells":{"boxes":[{left,top,width,height}...],"coordinateWidth":W,"coordinateHeight":H,"allowClassicalFallback":false}}
  // then "\n", then the raw image bytes (same layout as /api/cell-annotations). Without the header the body is the bare image
  // (legacy clients) and the classical grid is used exactly as before.
  app.post('/api/inspect/upload', async (req, res) => {
    try {
      const rawBody = req.body;
      if (!Buffer.isBuffer(rawBody) || rawBody.length === 0) {
        res.status(400).json({ error: 'Empty image payload uploaded.' });
        return;
      }

      const rawNameHeader = req.headers['x-file-name'];
      const fileName =
        typeof rawNameHeader === 'string' && rawNameHeader.trim()
          ? safeDecode(rawNameHeader)
          : `Inspection_${Date.now()}.jpg`;

      let imageBytes: Buffer = rawBody;
      let reviewedCells: ReviewedCellInput | undefined;
      if (req.headers['x-inspect-envelope'] === '1') {
        const separator = rawBody.indexOf(10);
        if (separator <= 0) { res.status(400).json({ error: 'Inspection envelope header line is missing.' }); return; }
        let envelope: { cells?: Partial<ReviewedCellInput> };
        try { envelope = JSON.parse(rawBody.subarray(0, separator).toString('utf8')); }
        catch { res.status(400).json({ error: 'Inspection envelope header is not valid JSON.' }); return; }
        imageBytes = rawBody.subarray(separator + 1);
        if (imageBytes.length === 0) { res.status(400).json({ error: 'Empty image payload uploaded.' }); return; }
        const cells = envelope?.cells;
        // `boxes` present (even empty) means "use the Cell Annotation set"; absent means legacy classical behaviour.
        if (cells && Array.isArray(cells.boxes)) {
          reviewedCells = {
            boxes: cells.boxes,
            coordinateWidth: cells.coordinateWidth,
            coordinateHeight: cells.coordinateHeight,
            allowClassicalFallback: cells.allowClassicalFallback === true,
          };
        }
      }

      const parsedHeaderConfig = readConfigHeader(req);

      const includePreview = req.headers['x-include-preview'] === 'true';
      const config = parseInspectionConfig(parsedHeaderConfig);
      const result = await analyzeInspectionImage(imageBytes, fileName, config, includePreview, { runDirectCellDetection: false, reviewedCells });
      res.json(result);
    } catch (err: unknown) {
      console.error('Upload inspection error:', err);
      const message = err instanceof Error ? err.message : 'Failed to analyze uploaded image.';
      res.status(500).json({ error: message });
    }
  });

  // Server-side batch ZIP extraction & GeoJSON Point aggregation endpoint.
  // Not used by the app UI. It has no Cell Annotation boxes to work from, so it keeps the classical cell grid (legacy API behaviour);
  // use /api/inspect/upload with the cell envelope when cells must come from the Cell Annotation stage.
  app.post('/api/inspect/upload-zip', async (req, res) => {
    try {
      const rawZip = req.body;
      if (!Buffer.isBuffer(rawZip) || rawZip.length === 0) {
        res.status(400).json({ error: 'Empty ZIP archive uploaded.' });
        return;
      }

      const config = parseInspectionConfig(readConfigHeader(req));

      const zip = await JSZip.loadAsync(rawZip);
      const imageEntries = Object.values(zip.files).filter(
        (entry) => !entry.dir && isSupportedImageEntry(entry.name)
      );

      if (imageEntries.length === 0) {
        res.status(400).json({
          error: 'No supported images (.jpg, .jpeg, .jfif, .png, .webp) found inside ZIP archive.',
        });
        return;
      }

      if (imageEntries.length > MAX_ZIP_ENTRIES) {
        res.status(413).json({ error: `ZIP has ${imageEntries.length} images; limit is ${MAX_ZIP_ENTRIES}.` });
        return;
      }

      const results: InspectionResult[] = [];
      let unpacked = 0;
      for (const entry of imageEntries) {
        const imgBuf = await entry.async('nodebuffer');
        unpacked += imgBuf.length; // zip-bomb guard
        if (unpacked > MAX_ZIP_UNPACKED_BYTES) {
          res.status(413).json({ error: 'ZIP unpacks beyond the allowed size.' });
          return;
        }
        const resItem = await analyzeInspectionImage(imgBuf, entry.name, config, false, { runDirectCellDetection: false }); // crack stage only; cell localization is a separate batch operation
        results.push(resItem);
      }

      const mergedGeoJson = mergeServerGeoJson(results, config.onlyImagesWithIssuesInGeoJson);
      res.json({
        framesCount: results.length,
        results,
        mergedGeoJson,
      });
    } catch (err: unknown) {
      console.error('ZIP batch inspection error:', err);
      const message = err instanceof Error ? err.message : 'Failed to process ZIP archive.';
      res.status(500).json({ error: message });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    const publicPath = path.resolve(__dirname, 'public');
    app.use(express.static(distPath));
    app.use(express.static(publicPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  // JSON/body-parser errors -> clean 4xx instead of HTML stack traces
  app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
    const status = (err as { status?: number })?.status ?? 500;
    if (res.headersSent) { _next(err); return; } // can't write a second response
    res.status(status).json({ error: err instanceof Error ? err.message : 'Server error' });
  });

  const HOST = BIND_HOST;
  // Optional: preload metadata via METADATA_JSON=/path/to/raw-images-meta-data.json
  if (process.env.METADATA_JSON) {
    try {
      console.log(`Metadata: ${loadMetadataFile(process.env.METADATA_JSON)} records loaded`);
    } catch (e) {
      console.warn('METADATA_JSON could not be loaded:', e instanceof Error ? e.message : e);
    }
  }

  const server = app.listen(PORT, HOST, () => {
    console.log(`Persistent user data: ${DATA_ROOT} (scanned ${DATA_MIGRATION_RESULT.scannedReleases} release folder(s); imported ${DATA_MIGRATION_RESULT.importedFiles} changed file(s)).`);
    console.log(`PV CrackVision running on http://localhost:${PORT}`);
    console.log('Engine: classical-offline - fully local, no network needed');
  });
  server.on('error', (error: NodeJS.ErrnoException) => {
    if (error.code === 'EADDRINUSE') console.error(`Port ${PORT} is already in use. Close the other PV CrackVision instance or set PORT to a free port.`);
    else console.error('HTTP server error:', error);
    // Inside the desktop app (server runs in Electron's main process) keep Electron's own error handling; standalone: exit non-zero.
    if (process.versions.electron) throw error;
    process.exit(1);
  });
  // Standalone (npm start / start.bat) only: the Electron shell owns lifecycle and crash logging for the desktop build.
  if (!process.versions.electron) {
    let shuttingDown = false;
    const shutdown = (signal: string) => {
      if (shuttingDown) return;
      shuttingDown = true;
      console.log(`${signal} received: finishing in-flight requests and closing the HTTP server...`);
      server.close(() => process.exit(0));
      server.closeIdleConnections?.();
      setTimeout(() => process.exit(0), 5000).unref(); // every data write is temp-file + rename, so a forced exit cannot leave a torn file
    };
    for (const signal of ['SIGINT', 'SIGTERM'] as const) process.on(signal, () => shutdown(signal));
    process.on('unhandledRejection', (reason) => console.error('Unhandled promise rejection (request continues):', reason));
    process.on('uncaughtException', (error) => { console.error('Fatal uncaught exception:', error); process.exit(1); });
  }
}

startServer().catch((e) => {
  console.error('Fatal startup error:', e);
  process.exit(1);
});
