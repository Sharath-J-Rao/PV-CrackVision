/** Import (load) trained models from disk into the active data folder. Works identically in a packaged .exe: models are data files, never bundled. */
import fs from 'node:fs';
import path from 'node:path';
import { MODEL_DATA_DIR, TRAINING_MODELS_DIR } from './dataPaths.ts';
import { resetCellDetectorSession } from './cellDetector.ts';

const ACTIVE_ONNX = path.join(MODEL_DATA_DIR, 'cell.onnx');
const PREVIOUS_ONNX = path.join(MODEL_DATA_DIR, 'cell.previous.onnx');
const CHECKPOINT = path.join(TRAINING_MODELS_DIR, 'cell-best.pt');
export const ACTIVE_CELL_MODEL_PATH = ACTIVE_ONNX;

export type ImportResult = { kind: 'cell-onnx' | 'crack-classifier' | 'training-checkpoint'; fileName: string; bytes: number; message: string; warning?: string };

function backupAndReplace(temporary: string, active: string, previous: string) {
  if (fs.existsSync(active)) fs.copyFileSync(active, previous);
  fs.renameSync(temporary, active);
}

async function validateOnnx(file: string): Promise<string | undefined> {
  try {
    const ort: any = await import('onnxruntime-node');
    const session = await (ort.InferenceSession ?? ort.default?.InferenceSession).create(file, { executionProviders: ['cpu'] });
    if (!session.inputNames?.length || !session.outputNames?.length) throw new Error('The model has no inputs/outputs.');
    return undefined;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (/Cannot find (module|package)|ERR_MODULE_NOT_FOUND/i.test(message)) return 'onnxruntime-node is not installed, so the model could not be test-loaded. It was imported on file-format checks only.';
    throw new Error(`This file is not a loadable ONNX model: ${message}`);
  }
}

export async function importModelFile(buffer: Buffer, rawName: string): Promise<ImportResult> {
  const fileName = path.basename(rawName || 'model').replace(/[^\w.\- ]/g, '_');
  const ext = path.extname(fileName).toLowerCase();
  if (!buffer.length) throw new Error('The uploaded file is empty.');
  fs.mkdirSync(MODEL_DATA_DIR, { recursive: true });
  if (ext === '.onnx') {
    if (buffer.length < 1024 || buffer[0] !== 0x08) throw new Error('This does not look like an ONNX model file.');
    const temporary = `${ACTIVE_ONNX}.import.tmp`;
    fs.writeFileSync(temporary, buffer);
    try {
      const warning = await validateOnnx(temporary);
      backupAndReplace(temporary, ACTIVE_ONNX, PREVIOUS_ONNX);
      resetCellDetectorSession();
      return { kind: 'cell-onnx', fileName, bytes: buffer.length, warning, message: 'Cell detector model loaded and activated. The previous model was kept for roll-back.' };
    } catch (error) { try { fs.unlinkSync(temporary); } catch { /* already gone */ } throw error; }
  }
  if (ext === '.json') throw new Error('The legacy crack-classifier JSON model is no longer supported (v1.4.15.1). Train the crop classifier in the Training Centre instead.');
  if (ext === '.pt') {
    if (buffer.length < 1024 || buffer[0] !== 0x50 || buffer[1] !== 0x4b) throw new Error('This does not look like a PyTorch checkpoint (.pt).');
    fs.mkdirSync(TRAINING_MODELS_DIR, { recursive: true });
    const temporary = `${CHECKPOINT}.import.tmp`;
    fs.writeFileSync(temporary, buffer);
    fs.renameSync(temporary, CHECKPOINT);
    return { kind: 'training-checkpoint', fileName, bytes: buffer.length, message: 'Training checkpoint loaded. The next training run will continue from it automatically.' };
  }
  throw new Error('Unsupported model file. Use .onnx (cell detector), .json (crack classifier) or .pt (training checkpoint).');
}
