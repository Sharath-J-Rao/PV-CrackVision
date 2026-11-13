/**
 * v1.4.14 job runner for the two crack models trained by training/train_crack_classifier.py:
 *   'cls' = option A (pretrained crop classifier, decides crack/clean)   'det' = option B (tiled detector, shadow only).
 * Each mode is an independent background process with its own state file, so A and B can train at the same time.
 * The worker writes the state file; this module only starts, reads, cancels and reloads models when a job finishes.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawn, execFile } from 'node:child_process';
import { DATA_ROOT, FEEDBACK_DIR, RUNTIME_DIR, RUNTIME_MARKER } from './dataPaths.ts';
import { pythonExecutable, trainingWorkerPath } from './cellTraining.ts';
import { listFeedback } from './feedbackStore.ts';
import { getTrainingSettings } from './trainingSettings.ts';
import { CLASSIFIER_META, CLASSIFIER_ONNX, activeClassifierFiles, getCellClassifierWarning, resetCellClassifierSession } from './cellClassifier.ts';
import { MODULE_META, MODULE_ONNX, resetModuleShadowSession } from './moduleShadow.ts';

export type ClassifierMode = 'cls' | 'det';
export const isClassifierMode = (value: unknown): value is ClassifierMode => value === 'cls' || value === 'det';

const ACTIVE = ['preparing', 'training', 'validating'];
const jobDir = (mode: ClassifierMode) => path.join(DATA_ROOT, 'training', 'crack_net', mode);
const stateFile = (mode: ClassifierMode) => path.join(jobDir(mode), 'job-state.json');
const children = new Map<ClassifierMode, ReturnType<typeof spawn>>();
const reloaded = new Map<ClassifierMode, string>(); // jobId whose finished model was already hot-reloaded

const pidAlive = (pid?: number): boolean => { if (!pid || pid < 1) return false; try { process.kill(pid, 0); return true; } catch (e) { return (e as NodeJS.ErrnoException).code === 'EPERM'; } };
const readState = (mode: ClassifierMode): Record<string, any> => { try { return JSON.parse(fs.readFileSync(stateFile(mode), 'utf8')); } catch { return { status: 'idle', progress: 0, message: 'No run yet.', log: [], metrics: [] }; } };
function patchState(mode: ClassifierMode, patch: Record<string, any>): void {
  fs.mkdirSync(jobDir(mode), { recursive: true });
  const tmp = `${stateFile(mode)}.node.tmp`;
  fs.writeFileSync(tmp, JSON.stringify({ ...readState(mode), ...patch, updatedAt: new Date().toISOString() }, null, 1));
  fs.renameSync(tmp, stateFile(mode));
}

/** Status for the UI. Also (a) marks a job whose worker died as failed and (b) hot-reloads the model once when a job completes. */
export function getClassifierTrainingStatus(mode: ClassifierMode) {
  const state = readState(mode);
  if (ACTIVE.includes(state.status) && state.pid && !pidAlive(state.pid) && !children.has(mode)) {
    patchState(mode, { status: 'failed', message: 'The training process ended before it finished; the previous model was kept.', error: state.error ?? 'Worker process is no longer running.', finishedAt: new Date().toISOString() });
    return { ...state, ...readState(mode) };
  }
  if (state.status === 'completed' && state.jobId && reloaded.get(mode) !== state.jobId) {
    reloaded.set(mode, state.jobId);
    if (mode === 'cls') resetCellClassifierSession(); else resetModuleShadowSession();
  }
  return state;
}

/** Labelled-cell counts that decide whether a run can start (same image-level needs as the legacy trainer, plus option-B needs). */
export function classifierReadiness(mode: ClassifierMode): { ready: boolean; missing: string[] } {
  const items = listFeedback().filter((item) => item.category === 'spider_web_crack');
  const positives = items.filter((i) => i.label === 1), negatives = items.filter((i) => i.label === 0);
  const images = new Set(items.map((i) => i.imageFile)), positiveImages = new Set(positives.map((i) => i.imageFile));
  const missing: string[] = [];
  if (images.size < 4) missing.push(`${images.size}/4 reviewed images`);
  if (positives.length < 8) missing.push(`${positives.length}/8 cracked cells`);
  if (positiveImages.size < 2) missing.push(`${positiveImages.size}/2 images with a cracked cell`);
  if (mode === 'cls' && negatives.length < 8) missing.push(`${negatives.length}/8 clean cells`);
  return { ready: missing.length === 0, missing };
}

export function startClassifierTraining(mode: ClassifierMode) {
  const current = getClassifierTrainingStatus(mode);
  if (ACTIVE.includes(current.status)) throw new Error(`The ${mode === 'cls' ? 'classifier' : 'module detector'} is already training.`);
  if (!fs.existsSync(RUNTIME_MARKER)) throw new Error('Set up the YOLO training engine first (Training Centre -> Direct cell detection -> Set up).');
  const readiness = classifierReadiness(mode);
  if (!readiness.ready) throw new Error(`Not enough reviewed labels: ${readiness.missing.join(', ')}.`);

  const s = getTrainingSettings().crackNet;
  const cls = mode === 'cls';
  const jobId = crypto.randomUUID();
  const work = path.join(jobDir(mode), 'work');
  fs.mkdirSync(jobDir(mode), { recursive: true });
  fs.writeFileSync(stateFile(mode), JSON.stringify({ status: 'preparing', mode, jobId, progress: 0, message: 'Starting…', log: [], metrics: [], startedAt: new Date().toISOString() }));
  const args = [trainingWorkerPath('train_crack_classifier.py'), '--mode', mode, '--feedback-dir', FEEDBACK_DIR, '--work', work, '--state-file', stateFile(mode),
    '--activate', cls ? CLASSIFIER_ONNX : MODULE_ONNX, '--meta-activate', cls ? CLASSIFIER_META : MODULE_META,
    '--base-model', cls ? s.clsBase : s.detBase, '--epochs', String(cls ? s.clsEpochs : s.detEpochs), '--imgsz', String(cls ? s.clsImgsz : s.detImgsz),
    '--batch', String(cls ? s.clsBatch : s.detBatch), '--patience', String(cls ? s.clsPatience : s.detPatience), '--seed', String(s.seed),
    '--val-fraction', String(s.valFraction), '--cell-buffer', String(getTrainingSettings().crackClassifier.cellBuffer),
    '--target-recall', String(s.targetRecall), '--min-balanced-accuracy', String(s.minBalancedAccuracy), '--tile-overlap', String(s.detTileOverlap),
    '--edge-weight', String(getTrainingSettings().crackClassifier.edgeSampleWeight), '--hard-negative-weight', String(getTrainingSettings().crackClassifier.hardNegativeWeight),
    '--missed-crack-weight', String(getTrainingSettings().crackClassifier.missedCrackWeight)];
  const logFd = fs.openSync(path.join(jobDir(mode), 'worker.log'), 'a');
  try {
    const child = spawn(pythonExecutable(), args, { cwd: DATA_ROOT, windowsHide: true, stdio: ['ignore', logFd, logFd],
      env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8', YOLO_CONFIG_DIR: path.join(RUNTIME_DIR, 'ultralytics_config') } });
    children.set(mode, child);
    patchState(mode, { pid: child.pid });
    child.unref();
    child.on('error', (error) => { children.delete(mode); patchState(mode, { status: 'failed', message: 'The training process could not start.', error: error.message, finishedAt: new Date().toISOString() }); });
    child.on('close', (code) => {
      children.delete(mode);
      const state = readState(mode);
      // The worker records its own outcome; only a crash that bypassed it (killed, out of memory) needs a failure record here.
      if (ACTIVE.includes(state.status)) patchState(mode, { status: 'failed', message: `The training process exited unexpectedly (code ${code}). The previous model was kept.`, finishedAt: new Date().toISOString() });
    });
  } finally { fs.closeSync(logFd); }
  return getClassifierTrainingStatus(mode);
}

/** Stops the worker (whole process tree on Windows, where Ultralytics data-loader workers would otherwise linger). */
export function cancelClassifierTraining(mode: ClassifierMode) {
  const state = readState(mode);
  if (ACTIVE.includes(state.status) && state.pid) {
    if (process.platform === 'win32') execFile('taskkill', ['/pid', String(state.pid), '/T', '/F'], () => undefined);
    else { try { process.kill(state.pid, 'SIGTERM'); } catch { /* already gone */ } }
    patchState(mode, { status: 'failed', message: 'Training was cancelled by the operator; the previous model was kept.', error: 'Cancelled.', finishedAt: new Date().toISOString() });
  }
  return readState(mode);
}

/** Installed model metadata for the UI (null = nothing trained yet). */
export function installedClassifierModels(): { cls: Record<string, any> | null; det: Record<string, any> | null; warning?: string } {
  const read = (file: string, onnx: string) => { try { return fs.existsSync(onnx) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null; } catch { return null; } };
  const active = activeClassifierFiles(); // user-trained pair, else the bundled one
  return { cls: read(active.meta, active.onnx), det: read(MODULE_META, MODULE_ONNX), warning: getCellClassifierWarning() };
}

export const classifierLogText = (mode: ClassifierMode): string => (readState(mode).log ?? []).join('\n');

/** True while either pretrained crack model is being trained (a clear must never run underneath a live worker). */
export const isAnyClassifierTrainingActive = (): boolean =>
  (['cls', 'det'] as ClassifierMode[]).some((mode) => children.has(mode) || ACTIVE.includes(getClassifierTrainingStatus(mode).status));

/** One snapshot for the Training Centre panel; shared by the status route and the clear route so both always return the same shape. */
export function classifierStatusSnapshot() {
  return { cls: getClassifierTrainingStatus('cls'), det: getClassifierTrainingStatus('det'), installed: installedClassifierModels(), readiness: { cls: classifierReadiness('cls'), det: classifierReadiness('det') } };
}

/**
 * v1.4.14.4: "Clear A + B models and training data".
 * Removes the installed pretrained crack models (A classifier + B detector, each .onnx and its .json meta) and everything the two
 * trainers generated from the reviewed labels: job state, logs, cropped/tiled datasets and Ultralytics run folders.
 * Deliberately NOT touched: the reviewed cracked/clean labels and their source images (the operator's annotation work, shared with the
 * legacy classifier - cleared by "Clear crack dataset + model"), the bundled read-only baseline models, the legacy boosted-tree/CNN model,
 * and the Python runtime / downloaded base weights. With A removed, detection falls back to the legacy model exactly as before A was trained.
 * Refuses while a job is running so a worker is never deleted from under itself.
 */
export function clearClassifierModelsAndData() {
  if (isAnyClassifierTrainingActive()) throw new Error('A or B is still training. Cancel it first, then clear.');
  // maxRetries/retryDelay: on Windows a model file can stay locked for a moment after its ONNX session is released.
  const remove = (target: string) => fs.rmSync(target, { recursive: true, force: true, maxRetries: 5, retryDelay: 150 });
  resetCellClassifierSession();
  resetModuleShadowSession(); // release the cached sessions first so the files are not held open
  for (const file of [CLASSIFIER_ONNX, CLASSIFIER_META, MODULE_ONNX, MODULE_META]) { remove(file); remove(`${file}.tmp`); }
  for (const mode of ['cls', 'det'] as ClassifierMode[]) remove(jobDir(mode));
  reloaded.clear(); // a later run reuses nothing from the cleared jobs
  return classifierStatusSnapshot();
}
