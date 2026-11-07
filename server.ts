import dotenv from 'dotenv';
import express from 'express';
import type { NextFunction, Request, Response } from 'express';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import JSZip from 'jszip';
import sharp from 'sharp';
import { detectCellBoxes, getCellDetectorStatus, setCellDetectorMode } from './server/cellDetector.ts';
import { listCellAnnotations, saveCellAnnotations, saveNegativeCellSample } from './server/cellAnnotationStore.ts';
import { scaleCellBoxesToSourceImage } from './server/cellAnnotationCoordinates.ts';
import { cancelCellYoloTraining, getCellTrainingLogPath, getCellTrainingMetricsCsv, getCellTrainingStatus, rollbackCellModel, setupCellTrainingEngine, startCellYoloTraining } from './server/cellTraining.ts';
import { analyzeInspectionImage } from './server/cvPipeline.ts';
import { loadMetadata, loadMetadataFile, metadataCount } from './server/metadataStore.ts';
import { addFeedbackMany, existingFeedbackForImage, clearFeedbackDataset, feedbackCounts, initializeCellFeedbackStore, listFeedback, removeFeedback } from './server/feedbackStore.ts';
import { resetUserModel, trainFromFeedback } from './server/trainer.ts';
import { addCrackEpochMetric, appendCrackTrainingLog, beginCrackTrainingJob, crackTrainingLogText, crackTrainingMetricsCsv, getCrackTrainingJobState, isCrackTrainingActive, updateCrackTrainingJob } from './server/crackClassifierTrainingJob.ts';
import { loadCrackModel } from './server/crackModel.ts';
import { DEFAULT_TRAINING_SETTINGS, getTrainingSettings, saveTrainingSettings } from './server/trainingSettings.ts';
import { learnProfileFromBoxes } from './server/cellTypeFilter.ts';
import type {
  GeoJsonFeatureCollection,
  GeoJsonPointFeature,
  InspectionConfig,
  InspectionResult,
} from './src/types/inspection.ts';
import { DATA_ROOT, migrateLegacyTrainingData } from './server/dataPaths.ts';
import { browseFolders, dataFolderUsage, describeDataFolder, resetDataFolderToDefault, setDataFolder } from './server/dataFolder.ts';
import { ACTIVE_CELL_MODEL_PATH, importModelFile } from './server/modelImport.ts';
import { isPackagedApp } from './server/appConfig.ts';

// .env.local first (per README), then .env
dotenv.config({ path: ['.env.local', '.env'] });

// Migrate legacy per-release runtime data before any request can read the shared stores.
const DATA_MIGRATION_RESULT = migrateLegacyTrainingData();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const MAX_ZIP_ENTRIES = 300;
const MAX_ZIP_UNPACKED_BYTES = 1.5 * 1024 * 1024 * 1024;

// v1.3.1+ uses reviewed cell crops; v1.4.1+ adds direct-cell YOLO training and an advanced-only annotation workspace.
// Legacy feedback schemas are preserved in timestamped backup folders during the one-time migration.
if (initializeCellFeedbackStore()) resetUserModel();

// Tiny fixed-window limiter so the API can't be hammered (each call may spend Gemini quota)
const RATE_LIMIT_PER_MIN = Number(process.env.RATE_LIMIT_PER_MIN) || 600;
const hits = new Map<string, { n: number; reset: number }>();
function rateLimit(req: Request, res: Response, next: NextFunction): void {
  const now = Date.now();
  const key = req.ip || 'local';
  const h = hits.get(key);
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

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000; // hosts inject PORT

  app.use('/api/inspect', rateLimit);
  app.use('/api/inspect/upload', express.raw({ type: '*/*', limit: '80mb' }));
  app.use('/api/inspect/upload-zip', express.raw({ type: '*/*', limit: '200mb' }));
  app.use(express.json({ limit: '20mb' }));

  app.get('/api/health', (_req, res) => {
    const key = process.env.GEMINI_API_KEY?.trim();
    const geminiConfigured =
      Boolean(key) && key !== 'MY_GEMINI_API_KEY' && process.env.CV_ENGINE !== 'classical';
    res.json({
      ok: true,
      engine: geminiConfigured ? 'gemini+classical' : 'classical-offline',
      geminiConfigured,
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
      const meta = JSON.parse(req.body.subarray(0, separator).toString('utf8')) as { fileName?: string; imageId?: string; coordinateWidth?: number; coordinateHeight?: number; boxes?: Array<{ left: number; top: number; width: number; height: number }> };
      const image = req.body.subarray(separator + 1);
      if (!image.length || !Array.isArray(meta.boxes) || meta.boxes.length < 1 || meta.boxes.length > 3000) { res.status(400).json({ error: 'Draw and save between 1 and 3,000 individual cell boxes.' }); return; }
      const normalized = await sharp(image).rotate().jpeg().toBuffer({ resolveWithObject: true });
      const imageWidth = normalized.info.width, imageHeight = normalized.info.height;
      if (!imageWidth || !imageHeight) { res.status(400).json({ error: 'Could not read source image dimensions.' }); return; }
      const boxes = scaleCellBoxesToSourceImage(meta.boxes, imageWidth, imageHeight, meta.coordinateWidth, meta.coordinateHeight);
      // Persist the exact inference source image (not an older training-set image)
      // with coordinates converted to its full, EXIF-normalized pixel space.
      const saved = saveCellAnnotations(normalized.data, typeof meta.fileName === 'string' ? path.basename(meta.fileName) : 'image', boxes, typeof meta.imageId === 'string' ? meta.imageId : undefined);
      const annotationRecords = listCellAnnotations();
      res.json({ saved: true, id: saved.id, sourceImageId: saved.sourceImageId, imageFile: saved.imageFile, fileName: saved.fileName, boxes: saved.boxes.length, images: annotationRecords.length, positiveImages: annotationRecords.filter((record) => record.boxes.length > 0).length });
    } catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not save cell annotations.' }); }
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
    const address = req.socket.remoteAddress ?? '';
    const loopback = address === '127.0.0.1' || address === '::1' || address === '::ffff:127.0.0.1';
    if (!loopback || req.get('x-pv-action') !== '1') { res.status(403).json({ error: 'This action is only available from the local PV CrackVision window.' }); return; }
    next();
  };
  let appVersion = 'unknown';
  try { appVersion = JSON.parse(fs.readFileSync(path.join(__dirname, 'package.json'), 'utf8')).version ?? appVersion; } catch { /* packaged builds may omit package.json */ }
  app.get('/api/app/info', (_req, res) => { res.setHeader('Cache-Control', 'no-store'); res.json({ version: appVersion, packaged: isPackagedApp(), dataFolder: describeDataFolder() }); });
  app.get('/api/data-folder/usage', localOnly, (_req, res) => res.json(dataFolderUsage()));
  app.get('/api/data-folder/browse', localOnly, (req, res) => res.json(browseFolders(typeof req.query.path === 'string' ? req.query.path : undefined)));
  app.post('/api/data-folder', localOnly, async (req, res) => {
    try {
      const result = await setDataFolder(String(req.body?.path ?? ''), { copyExisting: req.body?.copyExisting !== false, includeRuntime: req.body?.includeRuntime === true });
      res.json({ ...result, info: describeDataFolder() });
    } catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not set the data folder.' }); }
  });
  app.post('/api/data-folder/reset', localOnly, (_req, res) => { try { res.json({ ...resetDataFolderToDefault(), info: describeDataFolder() }); } catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not reset the data folder.' }); } });
  // Exit code 75 tells scripts/run-server.mjs (or the future desktop shell) to start the server again with the new data folder.
  app.post('/api/app/restart', localOnly, (_req, res) => { res.json({ restarting: true }); setTimeout(() => process.exit(75), 400); });
  app.post('/api/cell-model/import', localOnly, express.raw({ type: '*/*', limit: '600mb' }), async (req, res) => {
    try {
      const name = decodeURIComponent(String(req.get('x-file-name') ?? 'model'));
      res.json(await importModelFile(Buffer.isBuffer(req.body) ? req.body : Buffer.alloc(0), name));
    } catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not load the model.' }); }
  });
  app.get('/api/cell-model/export', localOnly, (_req, res) => {
    if (!fs.existsSync(ACTIVE_CELL_MODEL_PATH)) { res.status(404).json({ error: 'There is no active cell model to export yet.' }); return; }
    res.download(ACTIVE_CELL_MODEL_PATH, 'cell.onnx');
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
    if (['setting_up', 'training'].includes(getCellTrainingStatus().status)) { res.status(409).json({ error: 'A cell training operation is already running.' }); return; }
    void setupCellTrainingEngine().catch(() => undefined); res.json(getCellTrainingStatus());
  });
  app.post('/api/cell-training/start', async (req, res) => {
    try { if (req.body && typeof req.body === 'object') saveTrainingSettings({ ...getTrainingSettings(), cellDetector: { ...getTrainingSettings().cellDetector, ...req.body } }); res.json(await startCellYoloTraining(req.body && typeof req.body === 'object' ? req.body : {})); }
    catch (error) { res.status(400).json({ ...getCellTrainingStatus(), error: error instanceof Error ? error.message : 'Could not start cell training.' }); }
  });
  app.post('/api/cell-training/cancel', (_req, res) => { try { res.json(cancelCellYoloTraining()); } catch (error) { res.status(500).json({ error: error instanceof Error ? error.message : 'Cancel failed.' }); } });
  app.post('/api/cell-training/rollback', (_req, res) => { try { res.json(rollbackCellModel()); } catch (error) { res.status(400).json({ error: error instanceof Error ? error.message : 'Could not roll back the cell model.' }); } });

  // ---- Reviewed cell annotations + fresh cell-level model training ----
  const feedbackState = () => {
    const m = loadCrackModel();
    const modelAlgorithm = !m ? null : m.schema === 'cell-level-v1' ? 'logistic' : (m.algorithm ?? (m.cnn && m.gbm ? 'ensemble' : m.cnn ? 'cnn' : m.gbm ? 'gbm' : 'unknown'));
    return {
      ...feedbackCounts(), modelVersion: m?.version ?? null, userTuned: Boolean(m),
      modelAlgorithm, modelHasCnn: Boolean(m?.cnn), modelHasGbm: Boolean(m?.gbm),
      modelTrainedAt: m?.training?.trainedAt ?? null,
      modelValidation: m?.training?.validation ?? null,
      componentValidation: m?.training?.componentValidation ?? null,
    };
  };
  app.get('/api/feedback', (_req, res) => {
    try { res.json({ ...feedbackState(), items: listFeedback() }); } catch (err) { res.status(500).json({ error: err instanceof Error ? err.message : 'Could not read feedback.' }); }
  });
  app.post('/api/feedback/batch', express.raw({ type: '*/*', limit: '100mb' }), async (req, res) => {
    try {
      if (!Buffer.isBuffer(req.body) || req.body.length === 0) {
        res.status(400).json({ error: 'Empty image payload.' });
        return;
      }
      const separator = req.body.indexOf(10);
      if (separator <= 0) { res.status(400).json({ error: 'Batch metadata header is missing.' }); return; }
      const metadata = JSON.parse(req.body.subarray(0, separator).toString('utf8')) as {
        fileName?: string;
        coordinateWidth?: number;
        coordinateHeight?: number;
        replace?: boolean;
        annotations?: Array<{ left: number; top: number; width: number; height: number; label: 0 | 1; category?: string; moduleBox?: { left: number; top: number; width: number; height: number } }>;
      };
      const imageBytes = req.body.subarray(separator + 1);
      const annotations = metadata.annotations;
      if (!imageBytes.length || !Array.isArray(annotations) || annotations.length < 1 || annotations.length > 5000) {
        res.status(400).json({ error: 'Provide image data and between 1 and 5,000 cell annotations.' });
        return;
      }
      const fileName = typeof metadata.fileName === 'string' ? metadata.fileName : 'image';
      // v1.4.8.3: boxes arrive in the inspected frame (cell detection caps it at 8000 px); convert them
      // to the full EXIF-normalised source image the trainer reads, as cell-box saving already does.
      const dims = await sharp(imageBytes).rotate().metadata();
      const srcW = dims.width || 0, srcH = dims.height || 0;
      const hasSpace = Number.isFinite(metadata.coordinateWidth) && Number(metadata.coordinateWidth) > 0 && Number.isFinite(metadata.coordinateHeight) && Number(metadata.coordinateHeight) > 0;
      const kx = hasSpace && srcW ? srcW / Number(metadata.coordinateWidth) : 1;
      const ky = hasSpace && srcH ? srcH / Number(metadata.coordinateHeight) : 1;
      const metas = annotations.map((entry, index) => {
        const values = [entry.left, entry.top, entry.width, entry.height];
        if (values.some((n) => !Number.isFinite(n)) || entry.left < 0 || entry.top < 0 || entry.width <= 0 || entry.height <= 0 || (entry.label !== 0 && entry.label !== 1)) {
          throw new Error(`Annotation ${index + 1} has invalid coordinates or label.`);
        }
        const mb = entry.moduleBox;
        if (mb && [mb.left, mb.top, mb.width, mb.height].some((n) => !Number.isFinite(n)) || mb && (mb.left < 0 || mb.top < 0 || mb.width <= 0 || mb.height <= 0)) {
          throw new Error(`Annotation ${index + 1} has an invalid module outline.`);
        }
        return {
          fileName,
          left: Math.round(entry.left * kx),
          top: Math.round(entry.top * ky),
          width: Math.max(1, Math.round(entry.width * kx)),
          height: Math.max(1, Math.round(entry.height * ky)),
          label: entry.label,
          category: typeof entry.category === 'string' ? entry.category : 'spider_web_crack',
          ...(mb ? { moduleLeft: Math.round(mb.left * kx), moduleTop: Math.round(mb.top * ky), moduleWidth: Math.max(1, Math.round(mb.width * kx)), moduleHeight: Math.max(1, Math.round(mb.height * ky)) } : {}),
        };
      });
      const result = addFeedbackMany(imageBytes, metas, { replace: metadata.replace === true });
      const items = result.saved.map((saved, index) => ({ index, id: saved.id }));
      res.json({ items, replaced: result.replaced, ...feedbackState() });
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
  app.get('/api/train/status', (_req, res) => { res.setHeader('Cache-Control', 'no-store'); res.json(getCrackTrainingJobState()); });
  app.get('/api/train/metrics.csv', (_req, res) => { res.type('text/csv').attachment('crack-classifier-epoch-metrics.csv').send(crackTrainingMetricsCsv()); });
  app.get('/api/train/log', (_req, res) => { res.type('text/plain').attachment('crack-classifier-training.log').send(crackTrainingLogText()); });
  app.post('/api/train', (req, res) => {
    try {
      if (isCrackTrainingActive()) { res.status(409).json({ error: 'Crack-classifier training is already running.', ...getCrackTrainingJobState() }); return; }
      const current = getTrainingSettings();
      const settings = req.body && typeof req.body === 'object' ? saveTrainingSettings({ ...current, crackClassifier: { ...current.crackClassifier, ...req.body } }).crackClassifier : current.crackClassifier;
      const jobId = randomUUID();
      const initial = beginCrackTrainingJob(jobId, settings.algorithm, settings.cnnEpochs, `Queued ${settings.algorithm} classifier training.`);
      res.status(202).json(initial);
      void (async () => {
        try {
          appendCrackTrainingLog(`Dataset validation started. Algorithm=${settings.algorithm}; CNN epochs=${settings.cnnEpochs}; GBM trees=${settings.gbmTrees}.`);
          const report = await trainFromFeedback(settings, (event) => {
            const activeStatus = event.stage === 'preparing' ? 'preparing' : event.stage === 'validation' ? 'validating' : 'training';
            updateCrackTrainingJob({
              status: activeStatus, stage: event.stage, progress: event.progress, message: event.message,
              ...(event.epoch !== undefined ? { epoch: event.epoch } : {}),
              ...(event.epochs !== undefined ? { epochs: event.epochs } : {}),
              ...(event.tree !== undefined ? { tree: event.tree } : {}),
              ...(event.trees !== undefined ? { trees: event.trees } : {}),
              ...(event.samples !== undefined ? { samples: event.samples } : {}),
              ...(event.trainingSamples !== undefined ? { trainingSamples: event.trainingSamples } : {}),
              ...(event.validationSamples !== undefined ? { validationSamples: event.validationSamples } : {}),
            });
            appendCrackTrainingLog(event.message);
            if (event.metric) {
              addCrackEpochMetric({
                epoch: event.metric.epoch, trainLoss: event.metric.loss, validationLoss: event.metric.validationLoss,
                precision: event.metric.precision, recall: event.metric.recall, specificity: event.metric.specificity,
                balancedAccuracy: event.metric.balancedAccuracy, f1: event.metric.f1, component: 'CNN',
              });
              appendCrackTrainingLog(`CNN epoch ${event.metric.epoch}: train_loss=${event.metric.loss.toFixed(5)}; validation_loss=${event.metric.validationLoss.toFixed(5)}; precision=${event.metric.precision.toFixed(4)}; recall=${event.metric.recall.toFixed(4)}; specificity=${event.metric.specificity.toFixed(4)}; F1=${event.metric.f1.toFixed(4)}; balanced_accuracy=${event.metric.balancedAccuracy.toFixed(4)}.`);
            }
          });
          appendCrackTrainingLog(`Training finished. accepted=${report.accepted}; algorithm=${report.algorithm}; train samples=${report.samples}; held-out images=${report.validationImages}; held-out balanced accuracy=${report.validationAfter.balancedAccuracy}.`);
          updateCrackTrainingJob({ status: 'completed', stage: 'completed', progress: 100, message: report.accepted ? 'Held-out validation passed; the new crack classifier was activated.' : 'Training completed, but the candidate did not pass validation; the previous active model was retained.', finishedAt: new Date().toISOString(), pid: undefined, report: report as unknown as Record<string, any> });
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Crack-classifier training failed.';
          const stack = error instanceof Error ? error.stack ?? message : String(error);
          appendCrackTrainingLog(`ERROR: ${message}`);
          if (stack && stack !== message) appendCrackTrainingLog(stack);
          updateCrackTrainingJob({ status: 'failed', stage: 'failed', message: 'Crack-classifier training failed. See the saved error and log.', error: message, errorTraceback: stack, finishedAt: new Date().toISOString(), pid: undefined });
        }
      })();
    } catch (err: unknown) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Could not start crack-classifier training.' });
    }
  });
  app.delete('/api/train/reset', (_req, res) => {
    try { clearFeedbackDataset(); resetUserModel(); res.json(feedbackState()); } catch (err) { res.status(500).json({ error: err instanceof Error ? err.message : 'Reset failed.' }); } // fs errors (e.g. Windows locked files)
  });

  // Analyze single image frame (used for individual uploads, folder files, and client-streamed zip entries)
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

      const parsedHeaderConfig = readConfigHeader(req);

      const includePreview = req.headers['x-include-preview'] === 'true';
      const config = parseInspectionConfig(parsedHeaderConfig);
      const result = await analyzeInspectionImage(rawBody, fileName, config, includePreview, { runDirectCellDetection: false });
      res.json(result);
    } catch (err: unknown) {
      console.error('Upload inspection error:', err);
      const message = err instanceof Error ? err.message : 'Failed to analyze uploaded image.';
      res.status(500).json({ error: message });
    }
  });

  // Server-side batch ZIP extraction & GeoJSON Point aggregation endpoint
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

  const HOST = process.env.HOST || '127.0.0.1';
  // Optional: preload metadata via METADATA_JSON=/path/to/raw-images-meta-data.json
  if (process.env.METADATA_JSON) {
    try {
      console.log(`Metadata: ${loadMetadataFile(process.env.METADATA_JSON)} records loaded`);
    } catch (e) {
      console.warn('METADATA_JSON could not be loaded:', e instanceof Error ? e.message : e);
    }
  }

  app.listen(PORT, HOST, () => {
    const key = process.env.GEMINI_API_KEY?.trim();
    const online = Boolean(key) && key !== 'MY_GEMINI_API_KEY' && process.env.CV_ENGINE !== 'classical';
    console.log(`Persistent user data: ${DATA_ROOT} (scanned ${DATA_MIGRATION_RESULT.scannedReleases} release folder(s); imported ${DATA_MIGRATION_RESULT.importedFiles} changed file(s)).`);
    console.log(`PV CrackVision running on http://localhost:${PORT}`);
    console.log(
      online
        ? 'Engine: Gemini + classical'
        : 'Engine: classical-offline (no GEMINI_API_KEY) - fully local, no network needed'
    );
  });
}

startServer().catch((e) => {
  console.error('Fatal startup error:', e);
  process.exit(1);
});
