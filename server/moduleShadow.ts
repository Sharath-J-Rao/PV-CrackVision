/**
 * v1.4.14 option B (shadow): tiled YOLO crack detector trained by training/train_crack_classifier.py --mode det.
 * It never creates or hides a result. When enabled and a model is installed it runs beside the classifier and only tags each
 * classifier detection with `shadowAgree` (does the tiled detector also flag that cell?) so both can be compared on real images
 * before B is allowed to decide anything.
 */
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { MODEL_DATA_DIR } from './dataPaths.ts';
import { createOrtSession } from './cellClassifier.ts';
import { decodeYoloCellOutput } from './yoloOutputDecoder.ts';
import type { PixelBox } from '../src/types/inspection.ts';

export const MODULE_ONNX = path.join(MODEL_DATA_DIR, 'moduleCrackDetector.onnx');
export const MODULE_META = path.join(MODEL_DATA_DIR, 'moduleCrackDetector.json');

export interface ModuleDetectorMeta { schema: 'module-detector-v1'; version: string; imgsz: number; tileOverlap: number; confidence: number; iou: number; validation?: Record<string, number> }
export type ShadowBox = PixelBox & { confidence: number };

let cache: { key: string; value: { meta: ModuleDetectorMeta; session: any; ort: any } | null } | null = null;
export function resetModuleShadowSession(): void { cache = null; }

async function load() {
  let key: string;
  try { const a = fs.statSync(MODULE_ONNX), b = fs.statSync(MODULE_META); key = `${a.size}:${a.mtimeMs}:${b.mtimeMs}`; } catch { cache = null; return null; }
  if (cache?.key === key) return cache.value;
  let value: NonNullable<typeof cache>['value'] = null;
  try {
    const meta = JSON.parse(fs.readFileSync(MODULE_META, 'utf8')) as ModuleDetectorMeta;
    if (meta.schema === 'module-detector-v1' && Number.isInteger(meta.imgsz) && meta.imgsz >= 320) {
      const { session, ort } = await createOrtSession(MODULE_ONNX);
      value = { meta, session, ort };
    }
  } catch (error) { console.warn(`[Module shadow] could not load: ${error instanceof Error ? error.message : error}`); }
  cache = { key, value };
  return value;
}

export const isModuleShadowInstalled = (): boolean => fs.existsSync(MODULE_ONNX) && fs.existsSync(MODULE_META);

const origins = (size: number, tile: number, stride: number): number[] => {
  if (size <= tile) return [0];
  const xs: number[] = [];
  for (let v = 0; v + tile <= size; v += stride) xs.push(v);
  if (xs[xs.length - 1] !== size - tile) xs.push(size - tile); // flush last tile, same rule as the trainer
  return xs;
};

const iou = (a: PixelBox, b: PixelBox): number => {
  const w = Math.min(a.left + a.width, b.left + b.width) - Math.max(a.left, b.left), h = Math.min(a.top + a.height, b.top + b.height) - Math.max(a.top, b.top);
  if (w <= 0 || h <= 0) return 0;
  return (w * h) / (a.width * a.height + b.width * b.height - w * h);
};

/** Tiled native-resolution detection (no downscale), tile-border duplicates merged by NMS. Returns [] when no model is installed. */
export async function detectModuleCracks(rawRgb: Buffer, width: number, height: number, maxTiles = 400): Promise<ShadowBox[]> {
  const loaded = await load();
  if (!loaded) return [];
  const { meta, session, ort } = loaded;
  const tile = meta.imgsz, stride = Math.max(1, Math.floor(tile * (1 - meta.tileOverlap)));
  const xs = origins(width, tile, stride), ys = origins(height, tile, stride);
  if (xs.length * ys.length > maxTiles) return []; // bounded cost: a huge mosaic must not stall the crack run for a shadow model
  const all: ShadowBox[] = [];
  for (const y0 of ys) for (const x0 of xs) {
    const w = Math.min(tile, width - x0), h = Math.min(tile, height - y0);
    // Ultralytics letterboxes small tiles with grey 114; pad the same way so edge tiles are scored as in training.
    const { data } = await sharp(rawRgb, { raw: { width, height, channels: 3 } }).extract({ left: x0, top: y0, width: w, height: h })
      .extend({ right: tile - w, bottom: tile - h, background: { r: 114, g: 114, b: 114 } }).removeAlpha().raw().toBuffer({ resolveWithObject: true });
    const plane = tile * tile, input = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) { input[i] = data[i * 3] / 255; input[plane + i] = data[i * 3 + 1] / 255; input[2 * plane + i] = data[i * 3 + 2] / 255; }
    const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, tile, tile]) });
    const decoded = decodeYoloCellOutput(out[session.outputNames[0]], tile, tile, 1, 0, 0, meta.confidence, 300, { nmsIou: meta.iou, maxBoxes: 100, enclosingSuppression: false });
    for (const b of decoded.boxes) all.push({ left: b.left + x0, top: b.top + y0, width: b.width, height: b.height, confidence: b.confidence });
  }
  all.sort((a, b) => b.confidence - a.confidence);
  const kept: ShadowBox[] = [];
  for (const box of all) if (!kept.some((k) => iou(k, box) > meta.iou)) kept.push(box); // merge duplicates seen by overlapping tiles
  return kept;
}

/** True when any shadow box overlaps the cell (centre inside, or IoU > 0.1). */
export function shadowFlagsCell(shadow: ShadowBox[], cell: PixelBox): boolean {
  const cx = cell.left + cell.width / 2, cy = cell.top + cell.height / 2;
  return shadow.some((b) => (cx >= b.left && cx <= b.left + b.width && cy >= b.top && cy <= b.top + b.height) || iou(b, cell) > 0.1);
}
