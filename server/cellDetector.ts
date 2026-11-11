/** Optional YOLO11 ONNX cell detector (GPU when available, CPU fallback). Cell boundaries are always fitted separately. */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { resolveCellModel } from './modelResolver.ts';
import { decodeYoloCellOutput } from './yoloOutputDecoder.ts';
import { fuseOverlappingCellBoxes, suppressOversizedPatternOverlaps } from './cellDetectionGeometry.ts';
import { getTrainingSettings } from './trainingSettings.ts';
import { filterCellsByType } from './cellTypeFilter.ts';
import { buildCellInferenceTiles, offsetCellBoxes, type CellInferenceWindow } from './cellInferenceTiles.ts';
import { calculateLetterbox, DEFAULT_ONNX_IMAGE_SIZE, resolveOnnxInputShape } from './onnxInputShape.ts';
import sharp from 'sharp';
import type { PixelBox } from '../src/types/inspection.ts';

const SERVER_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(SERVER_DIR, '..');
const ONNX_RUNTIME_PACKAGE = path.join(ROOT, 'node_modules', 'onnxruntime-node', 'package.json');
export interface ScoredPixelBox extends PixelBox { confidence: number }
type DetectorMode = 'auto' | 'classical' | 'yolo11n';
let selectedMode: DetectorMode = 'auto';
let sessionPromise: Promise<any> | null = null;
let sessionFingerprint: string | null = null;
let loadedModelPath: string | null = null; // the candidate that actually loaded (user model, else the bundled one)
let activeProvider = 'not loaded yet'; // reported in status so the operator can see GPU vs CPU inference
// v1.4.12.7: set when a GPU provider loaded but failed while running; inference then stays on the CPU for this app session.
let gpuDisabled = false;
let gpuFailure: string | null = null;

export interface CellDetectionDiagnostics {
  imageWidth: number;
  imageHeight: number;
  modelInputWidth: number;
  modelInputHeight: number;
  inferencePasses: number;
  tilePasses: number;
  candidateCount: number;
  candidatesAboveThreshold: number;
  validBoxesBeforeMerge: number;
  boxesKept: number;
  maximumScore: number;
  confidenceThreshold: number;
  modelSizeBytes: number;
  modelModifiedAt: string;
  outputShape: string;
  candidatesLimited: number;
  rejectedByTypeFilter?: number; // v1.4.7.0: detections dropped by the cell-type/size filter
}


/** Call after a newly trained model is activated or a previous model is restored. */
export function resetCellDetectorSession(): void { sessionPromise = null; sessionFingerprint = null; }

export function setCellDetectorMode(mode: unknown): DetectorMode {
  if (mode === 'auto' || mode === 'classical' || mode === 'yolo11n') selectedMode = mode;
  return selectedMode;
}

export function getCellDetectorStatus() {
  const resolution = resolveCellModel();
  const modelExists = resolution.candidates.length > 0;
  const runtimeExists = fs.existsSync(ONNX_RUNTIME_PACKAGE);
  const yoloReady = modelExists && runtimeExists;
  return {
    selected: selectedMode,
    available: ['auto', 'classical', 'yolo11n'],
    modelExists,
    runtimeExists,
    modelPath: resolution.candidates[0]?.source === 'bundled' ? path.join('resources', 'models', 'cell.onnx') : path.join('data', 'models', 'cell.onnx'),
    modelSource: resolution.candidates[0]?.source ?? null,
    ...(resolution.warnings.length ? { modelWarnings: resolution.warnings } : {}),
    executionProvider: activeProvider,
    ...(gpuFailure ? { gpuFallback: gpuFailure } : {}),
    active: selectedMode === 'classical' ? 'classical' : yoloReady ? 'yolo11n' : selectedMode === 'yolo11n' ? 'unavailable' : 'classical',
    message: selectedMode === 'classical'
      ? (modelExists ? 'Classical cell-grid detection selected; YOLO11n cell weights are installed but not in use.' : 'Classical cell-grid detection selected; no YOLO11n weights are installed.')
      : !modelExists
        ? (selectedMode === 'yolo11n' ? 'YOLO11n was explicitly selected but data/models/cell.onnx is missing; no classical cell fallback will be run.' : 'No trained cell.onnx found; auto mode runs the classical cell-grid detector until YOLO weights are installed. Auto also falls back to it if YOLO fails or finds no cells.')
        : !runtimeExists
          ? 'Trained cell.onnx exists, but onnxruntime-node is missing. Relaunch with start.bat/start.sh to install the inference runtime; classical geometry is active for now.'
          : 'YOLO11n ONNX cell weights and the CPU inference runtime are available.',
  };
}

async function getSession(): Promise<any> {
  const candidates = resolveCellModel().candidates;
  if (!candidates.length) throw new Error('No valid cell.onnx is available (your own or the bundled model).');
  const stat = fs.statSync(candidates[0].path);
  const fingerprint = `${candidates[0].path}:${stat.size}:${stat.mtimeMs}`;
  // A shared model can be replaced by a different release or training process
  // while the Node process stays alive. Never keep serving a stale cached session.
  if (sessionFingerprint !== fingerprint) {
    sessionPromise = null;
    sessionFingerprint = fingerprint;
  }
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const runtimePackage = 'onnxruntime-node';
      const ort = await import(runtimePackage);
      // Prefer the GPU when this onnxruntime-node build ships one (DirectML on Windows, CUDA on Linux),
      // otherwise fall back to CPU. Why: a provider that is not installed makes create() throw, so each
      // attempt is isolated. PV_ONNX_PROVIDER=cpu forces CPU.
      const forced = process.env.PV_ONNX_PROVIDER?.trim().toLowerCase();
      const gpuFirst = gpuDisabled ? [] : process.platform === 'win32' ? ['dml', 'cuda'] : process.platform === 'linux' ? ['cuda'] : [];
      const attempts: string[][] = forced && forced !== 'auto' ? [[forced]] : [...gpuFirst.map((provider) => [provider, 'cpu']), ['cpu']];
      let lastError: unknown;
      // A model that passes the file checks but fails to load falls through to the next candidate (user -> bundled).
      for (const candidate of candidates) for (const executionProviders of attempts) {
        try {
          // DirectML does not support memory-pattern optimisation or parallel execution; ORT requires both to be off for it.
          const sessionOptions = { executionProviders, graphOptimizationLevel: 'all', ...(executionProviders[0] === 'dml' ? { enableMemPattern: false, executionMode: 'sequential' } : {}) };
          const session = await ort.InferenceSession.create(candidate.path, sessionOptions);
          loadedModelPath = candidate.path;
          activeProvider = executionProviders[0];
          console.log(`[Cell YOLO] ONNX execution provider: ${activeProvider}`);
          return session;
        } catch (error) { lastError = error; }
      }
      throw lastError;
    })().catch((error) => { sessionPromise = null; sessionFingerprint = null; throw error; });
  }
  return sessionPromise;
}

function iou(a: PixelBox, b: PixelBox): number {
  const x0 = Math.max(a.left, b.left), y0 = Math.max(a.top, b.top);
  const x1 = Math.min(a.left + a.width, b.left + b.width);
  const y1 = Math.min(a.top + a.height, b.top + b.height);
  const intersection = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  return intersection / Math.max(1, a.width * a.height + b.width * b.height - intersection);
}


/** Debug preview of exactly what one inference pass fed to the YOLO model (letterboxed, padded). */
export interface CellInputPreview {
  kind: 'full' | 'tile';
  window: { left: number; top: number; width: number; height: number };
  modelWidth: number; modelHeight: number;
  scale: number; padX: number; padY: number;
  dataUrl: string;
}
const MAX_INPUT_PREVIEWS = 6; // bound response size

export type CellDetectionOutcome = { boxes: ScoredPixelBox[]; source: 'yolo11n' | 'classical'; warning?: string; diagnostics?: CellDetectionDiagnostics; inputPreviews?: CellInputPreview[] };

async function detectWithYolo(rawRgb: Buffer, imageWidth: number, imageHeight: number, captureInputs = false): Promise<CellDetectionOutcome> {
  const inputPreviews: CellInputPreview[] = [];
  const status = getCellDetectorStatus();
  if (!status.modelExists) return selectedMode === 'yolo11n'
    ? { boxes: [], source: 'yolo11n', warning: 'YOLO11n was explicitly selected but data/models/cell.onnx is missing; no classical cell fallback was run.' }
    : { boxes: [], source: 'classical', warning: 'No trained cell.onnx is installed; auto mode used classical cell geometry.' };
  if (!status.runtimeExists) return { boxes: [], source: selectedMode === 'yolo11n' ? 'yolo11n' : 'classical', warning: selectedMode === 'yolo11n' ? 'YOLO11n was explicitly selected, but onnxruntime-node is not installed; no classical cell fallback was run. Relaunch through start.bat/start.sh to install the CPU runtime.' : 'Trained cell.onnx exists but onnxruntime-node is not installed. Relaunch through start.bat/start.sh to install the CPU runtime; auto mode used the classical detector because YOLO could not start.' };

  try {
    const session = await getSession();
    const inputName = session.inputNames[0];
    const inputMetadata = (session.inputMetadata ?? []).find((item: { name?: string }) => item.name === inputName);
    const inputShape = resolveOnnxInputShape(inputMetadata, DEFAULT_ONNX_IMAGE_SIZE);
    const modelStat = fs.statSync(loadedModelPath ?? resolveCellModel().candidates[0].path);
    // v1.4.7.0: every former constant now comes from the persisted, operator-editable Training Settings.
    const settings = getTrainingSettings();
    const inf = settings.cellInference;
    const windows: CellInferenceWindow[] = [
      { left: 0, top: 0, width: imageWidth, height: imageHeight, kind: 'full' },
      ...(inf.useTiles ? buildCellInferenceTiles(imageWidth, imageHeight, inputShape.width, inputShape.height, { maxTiles: inf.maxTiles, tileScale: inf.tileScale, overlap: inf.tileOverlap }) : []),
    ];
    // Localization favors recall by default; spurious boxes remain explicitly editable.
    const confidenceThreshold = inf.confidenceThreshold;
    const collected: ScoredPixelBox[] = [];
    let candidateCount = 0;
    let candidatesAboveThreshold = 0;
    let validBoxesBeforeMerge = 0;
    let maximumScore = 0;
    let outputShape = 'unknown';
    let candidatesLimited = 0;

    for (const window of windows) {
      const tileBuffer = window.left === 0 && window.top === 0 && window.width === imageWidth && window.height === imageHeight
        ? rawRgb
        : await sharp(rawRgb, { raw: { width: imageWidth, height: imageHeight, channels: 3 } })
          .extract({ left: window.left, top: window.top, width: window.width, height: window.height })
          .raw().toBuffer();
      const { scale, resizedWidth, resizedHeight, padX, padY } = calculateLetterbox(
        window.width, window.height, inputShape.width, inputShape.height,
      );
      const resized = await sharp(tileBuffer, { raw: { width: window.width, height: window.height, channels: 3 } })
        .resize(resizedWidth, resizedHeight, { fit: 'fill' })
        .extend({ top: padY, bottom: inputShape.height - resizedHeight - padY, left: padX, right: inputShape.width - resizedWidth - padX, background: { r: 114, g: 114, b: 114 } })
        .raw().toBuffer();
      // Capture the exact padded model input (pre-normalisation) so operators can see what YOLO sees.
      if (captureInputs && inputPreviews.length < MAX_INPUT_PREVIEWS) {
        try {
          const jpeg = await sharp(resized, { raw: { width: inputShape.width, height: inputShape.height, channels: 3 } }).jpeg({ quality: 70 }).toBuffer();
          inputPreviews.push({ kind: window.kind, window: { left: window.left, top: window.top, width: window.width, height: window.height }, modelWidth: inputShape.width, modelHeight: inputShape.height, scale, padX, padY, dataUrl: `data:image/jpeg;base64,${jpeg.toString('base64')}` });
        } catch { /* a failed debug preview must never break detection */ }
      }
      const plane = inputShape.width * inputShape.height;
      const chw = new Float32Array(3 * plane);
      for (let i = 0; i < plane; i++) {
        chw[i] = resized[i * 3] / 255;
        chw[plane + i] = resized[i * 3 + 1] / 255;
        chw[2 * plane + i] = resized[i * 3 + 2] / 255;
      }
      const runtimePackage = 'onnxruntime-node';
      const ort = await import(runtimePackage);
      const input = new ort.Tensor('float32', chw, [1, 3, inputShape.height, inputShape.width]);
      const outputs = await session.run({ [inputName]: input });
      const output = outputs[session.outputNames[0]];
      const decoded = decodeYoloCellOutput(output, window.width, window.height, scale, padX, padY, confidenceThreshold, inf.maxCandidatesBeforeNms, { nmsIou: inf.decoderNmsIou, maxBoxes: inf.maxDecoderBoxes, enclosingSuppression: inf.enclosingSuppression });
      outputShape = decoded.diagnostics.outputShape;
      candidatesLimited += decoded.diagnostics.candidatesLimited;
      candidateCount += decoded.diagnostics.candidates;
      candidatesAboveThreshold += decoded.diagnostics.candidatesAboveThreshold;
      validBoxesBeforeMerge += decoded.diagnostics.validBoxesBeforeSuppression;
      maximumScore = Math.max(maximumScore, decoded.diagnostics.maximumScore);
      collected.push(...offsetCellBoxes(decoded.boxes, window, imageWidth, imageHeight));
    }

    // The full-frame and overlapping-tile passes can propose the same cell.
    // Merge only geometrically overlapping boxes, retaining the highest score.
    const merged = suppressOversizedPatternOverlaps(inf.fuseTileBoxes ? fuseOverlappingCellBoxes(collected, inf.mergeNmsIou) : collected, inf.mergeNmsIou, inf.enclosingSuppression) // v1.4.12.8: fuse first, then the usual NMS/enclosure pass
      .sort((a, b) => b.confidence - a.confidence);
    // Only detections matching an enabled "accept" cell type (and no "reject" type) pass; the rest are rejected.
    const typed = filterCellsByType(merged, settings.cellTypeFilter);
    const boxes = typed.kept.slice(0, inf.maxFinalBoxes);
    const diagnostics: CellDetectionDiagnostics = {
      imageWidth, imageHeight,
      modelInputWidth: inputShape.width, modelInputHeight: inputShape.height,
      inferencePasses: windows.length,
      tilePasses: windows.length - 1,
      candidateCount, candidatesAboveThreshold, validBoxesBeforeMerge,
      boxesKept: boxes.length, maximumScore, confidenceThreshold,
      modelSizeBytes: modelStat.size,
      modelModifiedAt: modelStat.mtime.toISOString(),
      outputShape, candidatesLimited, rejectedByTypeFilter: typed.rejected.length,
    };
    console.log(`[Cell YOLO] image=${imageWidth}x${imageHeight}; model=${inputShape.width}x${inputShape.height}; passes=${windows.length} (tiles=${windows.length - 1}); candidates=${candidateCount}; above ${(confidenceThreshold * 100).toFixed(1)}%=${candidatesAboveThreshold}; valid=${validBoxesBeforeMerge}; topKLimited=${candidatesLimited}; kept=${boxes.length}; typeFilterRejected=${typed.rejected.length}; output=${outputShape}; maxScore=${(maximumScore * 100).toFixed(2)}%; modelBytes=${modelStat.size}; modelMtime=${modelStat.mtime.toISOString()}.`);
    const emptyWarning = boxes.length ? undefined
      : `YOLO ran the full image plus ${windows.length - 1} overlapping detail tile(s) using the active ${inputShape.width}×${inputShape.height} ONNX model, but kept no cell boxes (${typed.rejected.length} rejected by the cell-type filter). It evaluated ${candidateCount} raw candidates; ${candidatesAboveThreshold} reached ${(confidenceThreshold * 100).toFixed(2)}% confidence and the maximum score was ${(maximumScore * 100).toFixed(2)}%. This field image is not part of the existing training set unless you save a complete reviewed box set. Draw/correct all visible PV-cell boxes on this image, save it, then retrain to teach the model this image domain.`;
    return { boxes, source: 'yolo11n', diagnostics, ...(captureInputs ? { inputPreviews } : {}), ...(emptyWarning ? { warning: emptyWarning } : {}) };
  } catch (error) {
    // v1.4.12.7: a GPU provider can load yet fail while running (driver reset, unsupported operator). Retry once on the CPU so
    // detection keeps working; if the CPU fails too, the error was not GPU related, so keep the GPU enabled.
    if (activeProvider !== 'cpu' && activeProvider !== 'not loaded yet' && !gpuDisabled) {
      const failedProvider = activeProvider, reason = error instanceof Error ? error.message : String(error);
      gpuDisabled = true; sessionPromise = null; sessionFingerprint = null;
      const retry = await detectWithYolo(rawRgb, imageWidth, imageHeight, captureInputs);
      if (retry.warning?.startsWith('YOLO inference failed')) { gpuDisabled = false; sessionPromise = null; sessionFingerprint = null; }
      else { gpuFailure = `${failedProvider} failed (${reason.slice(0, 160)}); using the CPU`; console.warn(`[Cell YOLO] ${gpuFailure}`); }
      return retry;
    }
    return { boxes: [], source: 'yolo11n', warning: `YOLO inference failed${selectedMode === 'yolo11n' ? '; no classical cell fallback was run' : ''}. ${error instanceof Error ? error.message : 'Unknown inference error.'}` };
  }
}

// ---- v1.4.10: genuine classical fallback -------------------------------------------------------------------------
// cvPipeline registers its edge-periodicity cell-grid detector here (avoids a circular import).
export type ClassicalCellProvider = (rgb: Buffer, width: number, height: number) => Promise<ScoredPixelBox[]>;
let classicalProvider: ClassicalCellProvider | null = null;
export function registerClassicalCellProvider(provider: ClassicalCellProvider): void { classicalProvider = provider; }

async function runClassical(rgb: Buffer, width: number, height: number, reason: string, base?: CellDetectionOutcome): Promise<CellDetectionOutcome> {
  const extras = base ? { ...(base.diagnostics ? { diagnostics: base.diagnostics } : {}), ...(base.inputPreviews ? { inputPreviews: base.inputPreviews } : {}) } : {};
  if (!classicalProvider) return { boxes: [], source: 'classical', warning: `${reason}Classical cell-grid detector is not available.`, ...extras };
  try {
    const boxes = await classicalProvider(rgb, width, height);
    const note = boxes.length
      ? `Classical cell-grid detection found ${boxes.length} provisional cell(s); review and correct them before saving.`
      : 'Classical cell-grid detection also found no cells; draw the cells manually.';
    return { boxes, source: 'classical', warning: `${reason}${note}`, ...extras };
  } catch (error) {
    return { boxes: [], source: 'classical', warning: `${reason}Classical cell-grid detection failed: ${error instanceof Error ? error.message : 'unknown error'}.`, ...extras };
  }
}

/** auto = YOLO first, classical only when YOLO is unavailable, fails or finds nothing. yolo11n = YOLO only. classical = classical only. */
export async function detectCellBoxes(rawRgb: Buffer, imageWidth: number, imageHeight: number, captureInputs = false): Promise<CellDetectionOutcome> {
  if (selectedMode === 'classical') return runClassical(rawRgb, imageWidth, imageHeight, '');
  const yolo = await detectWithYolo(rawRgb, imageWidth, imageHeight, captureInputs);
  if (selectedMode === 'yolo11n' || yolo.boxes.length) return yolo;
  const reason = yolo.warning ? `${yolo.warning.split(/\.\s/)[0]}. ` : 'YOLO returned no cells. ';
  return runClassical(rawRgb, imageWidth, imageHeight, `Auto mode fell back to classical detection: ${reason}`, yolo);
}
