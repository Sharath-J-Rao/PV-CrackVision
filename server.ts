import dotenv from 'dotenv';
import express from 'express';
import type { NextFunction, Request, Response } from 'express';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import JSZip from 'jszip';
import { analyzeInspectionImage } from './server/cvPipeline.ts';
import { loadMetadata, loadMetadataFile, metadataCount } from './server/metadataStore.ts';
import { addFeedback, feedbackCounts, listFeedback, removeFeedback } from './server/feedbackStore.ts';
import { resetUserModel, trainFromFeedback } from './server/trainer.ts';
import { baseCrackModel, loadCrackModel } from './server/crackModel.ts';
import type {
  GeoJsonFeatureCollection,
  GeoJsonPointFeature,
  InspectionConfig,
  InspectionResult,
} from './src/types/inspection.ts';

// .env.local first (per README), then .env
dotenv.config({ path: ['.env.local', '.env'] });

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BUILTIN_SAMPLES: Record<string, { relPath: string; fileName: string }> = {
  'pcs011-right-edge': {
    relPath: 'public/assets/aistudio/image_ecd4bcb0.png',
    fileName: 'PCS011-R2-T11_RightEdge.png',
  },
  'edge-spiderweb-jfif': {
    relPath: 'public/assets/aistudio/f81d2698-4dfd-4e1a-b917-de5433d27e2a_848c4df5.jfif',
    fileName: 'f81d2698-4dfd-4e1a-b917-de5433d27e2a_848c4df5.jfif',
  },
  'dji-rtk-0074': {
    relPath: 'public/assets/aistudio/DJI_20261011114408_0074_Z_9902db27.JPG',
    fileName: 'DJI_20261011114408_0074_Z_9902db27.JPG',
  },
  'edge-closeup-png': {
    relPath: 'public/assets/aistudio/image_2e5da07a.png',
    fileName: 'image_2e5da07a.png',
  },
};

const resultCache = new Map<string, InspectionResult>();
const MAX_CACHE_ENTRIES = 50; // bounded (was unbounded)
const MAX_ZIP_ENTRIES = 300;
const MAX_ZIP_UNPACKED_BYTES = 1.5 * 1024 * 1024 * 1024;

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
      resultCache.clear(); // cached results used the old georeference
      res.json({ records: n });
    } catch (err: unknown) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Invalid metadata JSON.' });
    }
  });

  // ---- User feedback (add missed issues / delete false detections) + model fine-tuning ----
  const feedbackState = () => {
    const m = loadCrackModel();
    const base = baseCrackModel();
    return { ...feedbackCounts(), modelVersion: m?.version ?? null, userTuned: Boolean(m && base && m.version !== base.version) };
  };
  app.get('/api/feedback', (_req, res) => {
    res.json({ ...feedbackState(), items: listFeedback() });
  });
  app.post('/api/feedback', express.raw({ type: '*/*', limit: '80mb' }), (req, res) => {
    try {
      if (!Buffer.isBuffer(req.body) || req.body.length === 0) {
        res.status(400).json({ error: 'Empty image payload.' });
        return;
      }
      const label = req.headers['x-label'] === '1' ? 1 : req.headers['x-label'] === '0' ? 0 : null;
      const box = String(req.headers['x-box'] ?? '').split(',').map(Number);
      if (label === null || box.length !== 4 || box.some((n) => !Number.isFinite(n)) || box[2] <= 0 || box[3] <= 0) {
        res.status(400).json({ error: 'x-label (0|1) and x-box (left,top,width,height) are required.' });
        return;
      }
      const fileName = safeDecode(String(req.headers['x-file-name'] ?? 'image'));
      const category = safeDecode(String(req.headers['x-category'] ?? 'spider_web_crack'));
      const item = addFeedback(req.body, { fileName, left: box[0], top: box[1], width: box[2], height: box[3], label, category });
      res.json({ id: item.id, ...feedbackState() });
    } catch (err: unknown) {
      res.status(500).json({ error: err instanceof Error ? err.message : 'Could not save feedback.' });
    }
  });
  app.post('/api/feedback/batch', express.raw({ type: '*/*', limit: '80mb' }), (req, res) => {
    try {
      if (!Buffer.isBuffer(req.body) || req.body.length === 0) {
        res.status(400).json({ error: 'Empty image payload.' });
        return;
      }
      const separator = req.body.indexOf(10);
      if (separator <= 0) { res.status(400).json({ error: 'Batch metadata header is missing.' }); return; }
      const metadata = JSON.parse(req.body.subarray(0, separator).toString('utf8')) as {
        fileName?: string;
        annotations?: Array<{ left: number; top: number; width: number; height: number; label: 0 | 1; category?: string; moduleBox?: { left: number; top: number; width: number; height: number } }>;
      };
      const imageBytes = req.body.subarray(separator + 1);
      const annotations = metadata.annotations;
      if (!imageBytes.length || !Array.isArray(annotations) || annotations.length < 1 || annotations.length > 500) {
        res.status(400).json({ error: 'Provide image data and between 1 and 500 cell annotations.' });
        return;
      }
      const fileName = typeof metadata.fileName === 'string' ? metadata.fileName : 'image';
      const items = annotations.map((entry, index) => {
        const values = [entry.left, entry.top, entry.width, entry.height];
        if (values.some((n) => !Number.isFinite(n)) || entry.left < 0 || entry.top < 0 || entry.width <= 0 || entry.height <= 0 || (entry.label !== 0 && entry.label !== 1)) {
          throw new Error(`Annotation ${index + 1} has invalid coordinates or label.`);
        }
        const mb = entry.moduleBox;
        if (mb && [mb.left, mb.top, mb.width, mb.height].some((n) => !Number.isFinite(n)) || mb && (mb.left < 0 || mb.top < 0 || mb.width <= 0 || mb.height <= 0)) {
          throw new Error(`Annotation ${index + 1} has an invalid module outline.`);
        }
        const saved = addFeedback(imageBytes, {
          fileName,
          left: Math.round(entry.left),
          top: Math.round(entry.top),
          width: Math.round(entry.width),
          height: Math.round(entry.height),
          label: entry.label,
          category: typeof entry.category === 'string' ? entry.category : 'spider_web_crack',
          ...(mb ? { moduleLeft: Math.round(mb.left), moduleTop: Math.round(mb.top), moduleWidth: Math.round(mb.width), moduleHeight: Math.round(mb.height) } : {}),
        });
        return { index, id: saved.id };
      });
      res.json({ items, total: items.length, positives: annotations.filter((a) => a.label === 1).length, negatives: annotations.filter((a) => a.label === 0).length, ...feedbackState() });
    } catch (err: unknown) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Could not save cell feedback.' });
    }
  });
  app.delete('/api/feedback/:id', (req, res) => {
    res.json({ removed: removeFeedback(req.params.id), ...feedbackState() });
  });
  app.post('/api/train', async (_req, res) => {
    try {
      const report = await trainFromFeedback();
      resultCache.clear(); // cached sample results came from the previous model
      res.json({ ...report, ...feedbackState() });
    } catch (err: unknown) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Training failed.' });
    }
  });
  app.delete('/api/train/model', (_req, res) => {
    resetUserModel();
    resultCache.clear();
    res.json(feedbackState());
  });

  // Analyze any of the attached sample frames
  app.post('/api/inspect/sample', async (req, res) => {
    try {
      const sampleKey =
        typeof req.body?.sampleId === 'string' && BUILTIN_SAMPLES[req.body.sampleId]
          ? req.body.sampleId
          : 'edge-spiderweb-jfif';
      const sampleSpec = BUILTIN_SAMPLES[sampleKey];
      const config = parseInspectionConfig(req.body?.config);
      const forceRefresh = Boolean(req.body?.forceRefresh);
      const cacheKey = `${sampleKey}:${JSON.stringify(config)}`;

      if (!forceRefresh && resultCache.has(cacheKey)) {
        res.json(resultCache.get(cacheKey));
        return;
      }

      const sampleAbsPath = path.resolve(__dirname, sampleSpec.relPath);
      const buffer = await fs.readFile(sampleAbsPath);
      const result = await analyzeInspectionImage(buffer, sampleSpec.fileName, config, false);

      if (resultCache.size >= MAX_CACHE_ENTRIES) {
        resultCache.delete(resultCache.keys().next().value as string); // evict oldest
      }
      resultCache.set(cacheKey, result);
      res.json(result);
    } catch (err: unknown) {
      console.error('Sample inspection error:', err);
      const message = err instanceof Error ? err.message : 'Failed to run CV inspection pipeline.';
      res.status(500).json({ error: message });
    }
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
      const result = await analyzeInspectionImage(rawBody, fileName, config, includePreview);
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
        const resItem = await analyzeInspectionImage(imgBuf, entry.name, config, false); // no per-image previews: keeps response small
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
