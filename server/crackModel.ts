/** Cell-level crack classifier. A model exists only after the user trains it from reviewed cell crops. */
import fs from 'fs';
import sharp from 'sharp';
import path from 'path';
import { MODEL_DATA_DIR } from './dataPaths.ts';
import { resolveCrackModelFile } from './modelResolver.ts';

export interface CrackModel {
  schema: 'cell-level-v1' | 'cell-level-v2';
  version: string;
  window: number;
  stride: number;
  margin: number;
  /** v1.4.9: fraction of the cell short side added around the detected cell box when cropping (absent = 0, so older models score exactly as before). */
  cellBuffer?: number;
  featureNames: string[];
  mean: number[];
  scale: number[];
  coef: number[];
  intercept: number;
  threshold: number;
  /** v2 (v1.4.8.7): boosted trees on the richer features and/or a small CNN on the cell crop; probabilities are averaged. */
  algorithm?: 'gbm' | 'cnn' | 'ensemble';
  gbm?: GbmModel;
  cnn?: CnnModel;
  training?: { trainedAt?: string; algorithm?: string; samples?: number; sourceImages?: number; heldOutImages?: number; validation?: Record<string, number>; componentValidation?: Record<string, Record<string, number>> };
}

export { CELL_MODEL_SCHEMA, CELL_WINDOW_SIZE, CELL_WINDOW_MARGIN, CELL_FEATURE_NAMES } from './cellFeatures.ts';
import { cellCropFeatures, cellFeatureVector, CELL_MODEL_SCHEMA, CELL_MODEL_SCHEMA_V2, CELL_FEATURE_NAMES_V2, CELL_WINDOW_SIZE, CELL_WINDOW_MARGIN, type CellCropCapture } from './cellFeatures.ts';
import { gbmProbability, isValidGbm, type GbmModel } from './gradientBoosting.ts';
import { buildCnnInput, cnnProbability, isValidCnn, type CnnModel } from './cellCnn.ts';


/** Default minimal cell buffer (fraction of the short side); about one pixel on a 35 px cell, three on a 100 px cell. */
export const DEFAULT_CELL_BUFFER = 0.03;

/**
 * Crop rectangle for one detected cell: the box grown by a minimal buffer on every side (so the cell boundary is
 * included even when the detection is slightly tight), clamped to the image. buffer = 0 reproduces the exact box.
 */
export function bufferedCellCrop(
  imageWidth: number,
  imageHeight: number,
  box: { left: number; top: number; width: number; height: number },
  buffer = 0,
): { left: number; top: number; width: number; height: number } {
  const b = buffer > 0 ? Math.max(1, Math.round(Math.min(box.width, box.height) * buffer)) : 0;
  const left = Math.max(0, Math.floor(box.left - b));
  const top = Math.max(0, Math.floor(box.top - b));
  const right = Math.min(imageWidth, Math.ceil(box.left + box.width + b));
  const bottom = Math.min(imageHeight, Math.ceil(box.top + box.height + b));
  return { left, top, width: right - left, height: bottom - top };
}

let cached: CrackModel | null | undefined;
const MODEL_DIR = MODEL_DATA_DIR;
/** New schema-specific path deliberately cannot load or overwrite the legacy v1.3.0 window model. */
export const USER_MODEL_PATH = path.join(MODEL_DIR, 'cellCrackModel.user.json');

export function readModel(file: string): CrackModel | null {
  try {
    const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
    if (raw.schema === CELL_MODEL_SCHEMA_V2) {
      const featureCount = CELL_FEATURE_NAMES_V2.length;
      const hasGbm = raw.gbm !== undefined, hasCnn = raw.cnn !== undefined;
      return raw.featureNames?.length === featureCount && (hasGbm || hasCnn) &&
        (!hasGbm || isValidGbm(raw.gbm, featureCount)) && (!hasCnn || isValidCnn(raw.cnn)) &&
        Number.isFinite(raw.threshold) && raw.threshold > 0 && raw.threshold < 1 && raw.window > 0 && raw.margin >= 0
        ? (raw as CrackModel) : null;
    }
    const n = raw.featureNames?.length ?? 0;
    return raw.schema === CELL_MODEL_SCHEMA && n > 0 && raw.coef?.length === 2 * n &&
      raw.mean?.length === 2 * n && raw.scale?.length === 2 * n &&
      raw.scale.every((value: number) => Number.isFinite(value) && value > 0)
      ? (raw as CrackModel) : null;
  } catch { return null; }
}

/** No bundled/default classifier: inference uses the learned branch only after cell-level training. */
export function loadCrackModel(): CrackModel | null {
  // v1.4.13.1: the user's own classifier first, then the bundled one; an unreadable/invalid file is skipped.
  if (cached === undefined) cached = resolveCrackModelFile().candidates.reduce<CrackModel | null>((found, c) => found ?? readModel(c.path), null);
  return cached;
}

export function reloadCrackModel(): CrackModel | null {
  cached = undefined;
  return loadCrackModel();
}

export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));
export function modelProb(model: CrackModel, vec: number[]): number {
  let z = model.intercept;
  for (let j = 0; j < vec.length; j++) z += ((vec[j] - model.mean[j]) / model.scale[j]) * model.coef[j];
  return sigmoid(z);
}

/** The single shared feature path used by both user training and inference: one crop per cell. */
export async function extractCellFeatures(
  rawRgb: Buffer,
  imageWidth: number,
  imageHeight: number,
  box: { left: number; top: number; width: number; height: number },
  window = CELL_WINDOW_SIZE,
  margin = CELL_WINDOW_MARGIN,
): Promise<number[]> {
  const left = Math.max(0, Math.floor(box.left));
  const top = Math.max(0, Math.floor(box.top));
  const right = Math.min(imageWidth, Math.ceil(box.left + box.width));
  const bottom = Math.min(imageHeight, Math.ceil(box.top + box.height));
  const width = right - left, height = bottom - top;
  if (width < 8 || height < 8) throw new Error('Cell crop is outside the source image or too small.');
  const { data } = await sharp(rawRgb, { raw: { width: imageWidth, height: imageHeight, channels: 3 } })
    .extract({ left, top, width, height })
    .resize(window, window, { fit: 'fill', kernel: 'lanczos3' })
    .toColorspace('srgb')
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return cellCropFeatures(data, window, margin);
}

/** v2 component probabilities (boosted trees on rich features, CNN on the crop), one entry per component the model has. */
export function scoreV2Parts(model: CrackModel, features: ArrayLike<number>, cnnInput: Float32Array | null | undefined): number[] {
  const parts: number[] = [];
  if (model.gbm) parts.push(gbmProbability(model.gbm, features));
  if (model.cnn) {
    if (!cnnInput) throw new Error('This model needs the CNN crop input, but none was extracted.');
    parts.push(cnnProbability(model.cnn, cnnInput));
  }
  if (!parts.length) throw new Error('The classifier has no usable component.');
  return parts;
}

/** v2 probability: mean of the available components. */
export function scoreV2(model: CrackModel, features: ArrayLike<number>, cnnInput: Float32Array | null | undefined): number {
  const parts = scoreV2Parts(model, features, cnnInput);
  return parts.reduce((a, b) => a + b, 0) / parts.length;
}

/**
 * v1.4.12.9: score one cell and also return each component's own probability (a single entry for v1 / single-component models),
 * so callers can demand that every component agrees. `probability` is identical to scoreCell().
 */
export function scoreCellDetailed(model: CrackModel, features: number[], cnnInput?: Float32Array | null): { probability: number; parts: number[] } {
  if (model.schema !== CELL_MODEL_SCHEMA_V2) { const p = modelProb(model, cellFeatureVector(features)); return { probability: p, parts: [p] }; }
  const parts = scoreV2Parts(model, features, cnnInput);
  return { probability: parts.reduce((a, b) => a + b, 0) / parts.length, parts };
}

/** Score one cell with whichever schema the active model uses. `features` are the RAW extractor features. */
export function scoreCell(model: CrackModel, features: number[], cnnInput?: Float32Array | null): number {
  return scoreCellDetailed(model, features, cnnInput).probability;
}

export interface CellSample { features: number[]; cnn: Float32Array | null }

/**
 * Single shared extraction path for training AND inference (v1.4.8.7): one crop per cell, optional richer features and CNN input.
 * rich=false & cnn=false reproduces extractCellFeatures exactly.
 */
export async function extractCellSample(
  rawRgb: Buffer,
  imageWidth: number,
  imageHeight: number,
  box: { left: number; top: number; width: number; height: number },
  options: { window?: number; margin?: number; rich?: boolean; cnn?: boolean; buffer?: number } = {},
): Promise<CellSample> {
  const window = options.window ?? CELL_WINDOW_SIZE, margin = options.margin ?? CELL_WINDOW_MARGIN;
  const { left, top, width, height } = bufferedCellCrop(imageWidth, imageHeight, box, options.buffer ?? 0);
  if (width < 8 || height < 8) throw new Error('Cell crop is outside the source image or too small.');
  const { data } = await sharp(rawRgb, { raw: { width: imageWidth, height: imageHeight, channels: 3 } })
    .extract({ left, top, width, height })
    .resize(window, window, { fit: 'fill', kernel: 'lanczos3' })
    .toColorspace('srgb')
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const capture = options.cnn ? ({} as CellCropCapture) : undefined;
  const features = cellCropFeatures(data, window, margin, Boolean(options.rich), capture);
  return { features, cnn: capture ? buildCnnInput(capture) : null };
}

export type { GbmModel, CnnModel };
