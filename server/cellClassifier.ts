/**
 * Per-cell crack scoring models, both exported to ONNX with a [N,2] output = [1-p, p] so one scoring path serves them:
 *  - 'anomaly' (default since v1.4.17): clean-cell memory-bank anomaly detector, trained by training/train_crack_anomaly.py. p = anomaly score (0.5 = trained threshold).
 *  - 'classifier' (archived): pretrained cell-crop classifier (yolo11n-cls), trained by training/train_crack_classifier.py --mode cls. p = P(cracked).
 * One crop per detected cell.
 *
 * Preprocessing MUST match the trainer: crop = detected cell box + `cellBuffer`, squashed (fit:'fill') to imgsz x imgsz,
 * RGB, 0..1, NCHW. The model is "present" only when BOTH the .onnx and its .json meta exist and agree; the trainer writes the
 * meta last, so a half-activated model is never used.
 */
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { MODEL_DATA_DIR } from './dataPaths.ts';
import { bufferedCellCrop } from './crackModel.ts';
import { resolveAnomalyFiles, resolveClassifierFiles } from './modelResolver.ts';

export const CLASSIFIER_ONNX = path.join(MODEL_DATA_DIR, 'cellClassifier.onnx');
export const CLASSIFIER_META = path.join(MODEL_DATA_DIR, 'cellClassifier.json');
export const ANOMALY_ONNX = path.join(MODEL_DATA_DIR, 'cellAnomaly.onnx');
export const ANOMALY_META = path.join(MODEL_DATA_DIR, 'cellAnomaly.json');

export type CrackEngine = 'anomaly' | 'classifier';
const SCHEMAS: Record<CrackEngine, string> = { anomaly: 'cell-anomaly-v1', classifier: 'cell-classifier-v1' };

export interface CellClassifierMeta {
  schema: 'cell-classifier-v1' | 'cell-anomaly-v1';
  version: string;
  imgsz: number;
  classNames: string[];
  crackedIndex: number;
  /** Decision threshold picked on held-out images at the target recall. */
  threshold: number;
  cellBuffer: number;
  baseModel?: string;
  trainedAt?: string;
  validation?: Record<string, number>;
  msPerCell?: number;
  /** Anomaly models build an N x patches x bank distance matrix per batch, so they ask for a smaller batch. */
  batchSize?: number;
}
export interface LoadedCellClassifier { meta: CellClassifierMeta; session: any; provider: string; file: string }

/** GPU first when the onnxruntime-node build ships one, otherwise CPU; each attempt is isolated because a missing provider makes create() throw. */
export async function createOrtSession(file: string, forceCpu = false): Promise<{ session: any; provider: string; ort: any }> {
  const runtimePackage = 'onnxruntime-node'; // dynamic: the app must still start when the runtime is not installed
  const ort = await import(runtimePackage);
  const forced = process.env.PV_ONNX_PROVIDER?.trim().toLowerCase();
  const gpu = forceCpu ? [] : process.platform === 'win32' ? ['dml', 'cuda'] : process.platform === 'linux' ? ['cuda'] : [];
  const attempts: string[][] = forced && forced !== 'auto' ? [[forced]] : [...gpu.map((p) => [p, 'cpu']), ['cpu']];
  let lastError: unknown; const skipped: string[] = [];
  for (const executionProviders of attempts) {
    try {
      const options = { executionProviders, graphOptimizationLevel: 'all', ...(executionProviders[0] === 'dml' ? { enableMemPattern: false, executionMode: 'sequential' } : {}) };
      const session = await ort.InferenceSession.create(file, options);
      if (skipped.length) console.log(`[Crack model] GPU provider(s) unavailable, using ${executionProviders[0]}: ${skipped.join(' | ')}`);
      return { session, provider: executionProviders[0], ort };
    } catch (error) { lastError = error; skipped.push(`${executionProviders[0]}: ${error instanceof Error ? error.message.split('\n')[0] : String(error)}`); }
  }
  throw lastError;
}

/** Files in use: both engines = the user's trained pair, else the pair bundled with the app (read-only; anomaly since v1.4.17.6). */
export const activeClassifierFiles = (engine: CrackEngine = 'classifier'): { onnx: string; meta: string } =>
  engine === 'anomaly' ? resolveAnomalyFiles() ?? { onnx: ANOMALY_ONNX, meta: ANOMALY_META } : resolveClassifierFiles() ?? { onnx: CLASSIFIER_ONNX, meta: CLASSIFIER_META };
const fingerprint = (engine: CrackEngine): string | null => {
  try { const f = activeClassifierFiles(engine), a = fs.statSync(f.onnx), b = fs.statSync(f.meta); return `${f.onnx}:${a.size}:${a.mtimeMs}:${b.mtimeMs}`; } catch { return null; }
};
const cache = new Map<CrackEngine, { key: string; value: LoadedCellClassifier | null }>();
const warnings = new Map<CrackEngine, string | undefined>();
export const getCellClassifierWarning = (engine: CrackEngine = 'classifier'): string | undefined => warnings.get(engine);
export function resetCellClassifierSession(): void { cache.clear(); }
/** v1.4.17.7: which device each crack model runs on ('dml' / 'cuda' = GPU, 'cpu', or null = not loaded yet), for the UI and /api/classifier/status. */
export function getCrackInferenceDevices(): Record<CrackEngine, { provider: string | null; gpu: boolean; version?: string }> {
  const info = (engine: CrackEngine) => { const v = cache.get(engine)?.value; return { provider: v?.provider ?? null, gpu: Boolean(v && v.provider !== 'cpu'), version: v?.meta.version }; };
  return { anomaly: info('anomaly'), classifier: info('classifier') };
}

/** Validate meta before trusting it: a wrong imgsz/index would silently produce garbage probabilities. */
function readMeta(engine: CrackEngine): CellClassifierMeta | null {
  try {
    const raw = JSON.parse(fs.readFileSync(activeClassifierFiles(engine).meta, 'utf8'));
    const ok = raw?.schema === SCHEMAS[engine] && Number.isInteger(raw.imgsz) && raw.imgsz >= 32 && raw.imgsz <= 1024 &&
      Number.isInteger(raw.crackedIndex) && raw.crackedIndex >= 0 && Number.isFinite(raw.threshold) && raw.threshold > 0 && raw.threshold < 1 &&
      Number.isFinite(raw.cellBuffer) && raw.cellBuffer >= 0 && raw.cellBuffer <= 0.5 && (raw.batchSize === undefined || (Number.isInteger(raw.batchSize) && raw.batchSize >= 1 && raw.batchSize <= 256));
    return ok ? (raw as CellClassifierMeta) : null;
  } catch { return null; }
}

/** The trained model of `engine`, or null when none is installed / it cannot load (there is no fallback between engines; callers must report the missing model). */
export async function loadCellClassifier(engine: CrackEngine = 'classifier'): Promise<LoadedCellClassifier | null> {
  const key = fingerprint(engine);
  if (!key) { cache.delete(engine); return null; }
  const cached = cache.get(engine);
  if (cached?.key === key) return cached.value;
  let value: LoadedCellClassifier | null = null;
  const meta = readMeta(engine);
  if (!meta) warnings.set(engine, `${path.basename(activeClassifierFiles(engine).meta)} is missing or invalid; the ${engine} model was ignored.`);
  else {
    try { const file = activeClassifierFiles(engine).onnx; const { session, provider } = await createOrtSession(file); value = { meta, session, provider, file }; warnings.set(engine, undefined); console.log(`[Crack ${engine}] ${meta.version} on ${provider}`); }
    catch (error) { warnings.set(engine, `The ${engine} model could not load: ${error instanceof Error ? error.message : String(error)}`); }
  }
  cache.set(engine, { key, value });
  return value;
}

/** Ultralytics exports softmax probabilities; apply softmax only if the rows are not already a distribution. */
export function toProbabilities(scores: ArrayLike<number>, rows: number, classes: number): Float32Array {
  const out = new Float32Array(rows * classes);
  for (let r = 0; r < rows; r++) {
    let sum = 0, min = Infinity, max = -Infinity;
    for (let c = 0; c < classes; c++) { const v = scores[r * classes + c]; sum += v; min = Math.min(min, v); max = Math.max(max, v); }
    const isDistribution = min >= 0 && Math.abs(sum - 1) < 1e-3;
    let denom = 0;
    for (let c = 0; c < classes; c++) { const e = isDistribution ? scores[r * classes + c] : Math.exp(scores[r * classes + c] - max); out[r * classes + c] = e; denom += e; }
    for (let c = 0; c < classes; c++) out[r * classes + c] /= denom || 1;
  }
  return out;
}

/** One cell crop as a CHW float tensor (RGB, 0..1), identical to the trainer's PNG -> numpy path. */
async function cellTensor(rawRgb: Buffer, width: number, height: number, box: { left: number; top: number; width: number; height: number }, meta: CellClassifierMeta): Promise<Float32Array | null> {
  const crop = bufferedCellCrop(width, height, box, meta.cellBuffer);
  if (crop.width < 8 || crop.height < 8) return null;
  const { data } = await sharp(rawRgb, { raw: { width, height, channels: 3 } })
    .extract(crop).resize(meta.imgsz, meta.imgsz, { fit: 'fill', kernel: 'lanczos3' }).toColorspace('srgb').removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const plane = meta.imgsz * meta.imgsz, out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) { out[i] = data[i * 3] / 255; out[plane + i] = data[i * 3 + 1] / 255; out[2 * plane + i] = data[i * 3 + 2] / 255; }
  return out;
}

export interface CellView { flipH: boolean; flipV: boolean }
const IDENTITY: CellView[] = [{ flipH: false, flipV: false }];
/** Flips used for candidate re-scoring. The trainer augments with fliplr/flipud 0.5, so these views are in-distribution. */
export const TTA_VIEWS: CellView[] = [{ flipH: true, flipV: false }, { flipH: false, flipV: true }, { flipH: true, flipV: true }];

/** Flip a CHW float tensor of size x size. Equivalent to flipping the crop before resizing (Lanczos is symmetric), without re-cropping. */
export function flipTensor(src: Float32Array, size: number, view: CellView): Float32Array {
  if (!view.flipH && !view.flipV) return src;
  const out = new Float32Array(src.length), plane = size * size;
  for (let c = 0; c < 3; c++) for (let y = 0; y < size; y++) {
    const sy = view.flipV ? size - 1 - y : y, from = c * plane + sy * size, to = c * plane + y * size;
    if (view.flipH) for (let x = 0; x < size; x++) out[to + x] = src[from + size - 1 - x]; else out.set(src.subarray(from, from + size), to);
  }
  return out;
}

/**
 * P(cracked) per box and per view ([box][view]); -1 marks a cell whose crop was unusable (too small / outside the image).
 * Efficiency: the next batch's crops are prepared while ONNX Runtime scores the current one (crop work runs in sharp's thread pool, inference
 * in ORT's), and every view of a batch reuses the same crops. A GPU runtime failure retries once on the CPU.
 */
export async function classifyCellViews(
  rawRgb: Buffer, width: number, height: number,
  boxes: Array<{ left: number; top: number; width: number; height: number }>,
  classifier: LoadedCellClassifier, views: CellView[] = IDENTITY, batchSize = 32,
): Promise<number[][]> {
  const { meta } = classifier;
  batchSize = meta.batchSize ?? batchSize;
  const result = boxes.map(() => views.map(() => -1));
  const plane = 3 * meta.imgsz * meta.imgsz;
  const prepare = (offset: number) => Promise.all(boxes.slice(offset, offset + batchSize).map((box) => cellTensor(rawRgb, width, height, box, meta).catch(() => null)));
  let pending = boxes.length ? prepare(0) : null;
  for (let offset = 0; offset < boxes.length; offset += batchSize) {
    const tensors = await pending!;
    pending = offset + batchSize < boxes.length ? prepare(offset + batchSize) : null; // overlaps with the inference below
    const valid = tensors.map((t, i) => (t ? i : -1)).filter((i) => i >= 0);
    if (!valid.length) continue;
    for (let v = 0; v < views.length; v++) {
      const input = new Float32Array(valid.length * plane);
      valid.forEach((i, n) => input.set(flipTensor(tensors[i]!, meta.imgsz, views[v]), n * plane));
      const run = async (c: LoadedCellClassifier) => {
        const ort = await import('onnxruntime-node' as string);
        return c.session.run({ [c.session.inputNames[0]]: new ort.Tensor('float32', input, [valid.length, 3, meta.imgsz, meta.imgsz]) });
      };
      let outputs: any;
      try { outputs = await run(classifier); }
      catch (error) {
        if (classifier.provider === 'cpu') throw error;
        console.warn(`[Cell classifier] ${classifier.provider} failed (${error instanceof Error ? error.message : error}); retrying on CPU.`);
        const cpu = await createOrtSession(classifier.file, true);
        classifier.session = cpu.session; classifier.provider = cpu.provider; // keep the working session for the rest of the run
        outputs = await run(classifier);
      }
      const tensor = outputs[classifier.session.outputNames[0]];
      const classes = tensor.data.length / valid.length;
      if (!Number.isInteger(classes) || meta.crackedIndex >= classes) throw new Error(`Unexpected classifier output size ${tensor.data.length} for ${valid.length} crops.`);
      const probs = toProbabilities(tensor.data, valid.length, classes);
      valid.forEach((i, n) => { result[offset + i][v] = probs[n * classes + meta.crackedIndex]; });
    }
  }
  return result;
}

/** P(cracked) for every box, same order as `boxes`; -1 marks an unusable crop. Single (unflipped) view. */
export async function classifyCells(
  rawRgb: Buffer, width: number, height: number,
  boxes: Array<{ left: number; top: number; width: number; height: number }>,
  classifier: LoadedCellClassifier, batchSize = 32,
): Promise<number[]> {
  return (await classifyCellViews(rawRgb, width, height, boxes, classifier, IDENTITY, batchSize)).map((row) => row[0]);
}
