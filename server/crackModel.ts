/** Cell-level crack classifier. A model exists only after the user trains it from reviewed cell crops. */
import fs from 'fs';
import sharp from 'sharp';
import path from 'path';
import { MODEL_DATA_DIR } from './dataPaths.ts';

export interface CrackModel {
  schema: 'cell-level-v1' | 'cell-level-v2';
  version: string;
  window: number;
  stride: number;
  margin: number;
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
  if (cached === undefined) cached = readModel(USER_MODEL_PATH);
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

/** v2 probability: mean of the available components (boosted trees on rich features, CNN on the crop). */
export function scoreV2(model: CrackModel, features: ArrayLike<number>, cnnInput: Float32Array | null | undefined): number {
  const parts: number[] = [];
  if (model.gbm) parts.push(gbmProbability(model.gbm, features));
  if (model.cnn) {
    if (!cnnInput) throw new Error('This model needs the CNN crop input, but none was extracted.');
    parts.push(cnnProbability(model.cnn, cnnInput));
  }
  if (!parts.length) throw new Error('The classifier has no usable component.');
  return parts.reduce((a, b) => a + b, 0) / parts.length;
}

/** Score one cell with whichever schema the active model uses. `features` are the RAW extractor features. */
export function scoreCell(model: CrackModel, features: number[], cnnInput?: Float32Array | null): number {
  return model.schema === CELL_MODEL_SCHEMA_V2 ? scoreV2(model, features, cnnInput) : modelProb(model, cellFeatureVector(features));
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
  options: { window?: number; margin?: number; rich?: boolean; cnn?: boolean } = {},
): Promise<CellSample> {
  const window = options.window ?? CELL_WINDOW_SIZE, margin = options.margin ?? CELL_WINDOW_MARGIN;
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
  const capture = options.cnn ? ({} as CellCropCapture) : undefined;
  const features = cellCropFeatures(data, window, margin, Boolean(options.rich), capture);
  return { features, cnn: capture ? buildCnnInput(capture) : null };
}

export type { GbmModel, CnnModel };
