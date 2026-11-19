/**
 * Job runner for the crack models:
 *   'anom' = anomaly detector (default engine, training/train_crack_anomaly.py)
 *   'cls' = archived option A (pretrained crop classifier)   'det' = archived option B (tiled detector, shadow only), both training/train_crack_classifier.py.
 * Each mode is an independent background process with its own state file, so all of them can train at the same time.
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
import { bundledAnomalyFiles, resolveAnomalyFiles } from './modelResolver.ts';
import { ANOMALY_META, ANOMALY_ONNX, CLASSIFIER_META, CLASSIFIER_ONNX, activeClassifierFiles, getCellClassifierWarning, getCrackInferenceDevices, resetCellClassifierSession } from './cellClassifier.ts';
import { MODULE_META, MODULE_ONNX, resetModuleShadowSession } from './moduleShadow.ts';

export type ClassifierMode = 'cls' | 'det' | 'anom';
export const isClassifierMode = (value: unknown): value is ClassifierMode => value === 'cls' || value === 'det' || value === 'anom';
const MODE_NAME: Record<ClassifierMode, string> = { cls: 'classifier', det: 'module detector', anom: 'anomaly detector' };
const ARCHIVED: ClassifierMode[] = ['cls', 'det'];

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
    if (mode === 'det') resetModuleShadowSession(); else resetCellClassifierSession();
  }
  return state;
}

/** Labelled-cell counts that decide whether a run can start (same image-level needs as the legacy trainer, plus option-B needs). */
export function classifierReadiness(mode: ClassifierMode): { ready: boolean; missing: string[]; warnings: string[]; counts: { positives: number; negatives: number; images: number; positiveImages: number } } {
  const items = listFeedback().filter((item) => item.category === 'spider_web_crack');
  const positives = items.filter((i) => i.label === 1), negatives = items.filter((i) => i.label === 0);
  const images = new Set(items.map((i) => i.imageFile)), positiveImages = new Set(positives.map((i) => i.imageFile));
  const missing: string[] = [];
  const warnings: string[] = [];
  const counts = { positives: positives.length, negatives: negatives.length, images: images.size, positiveImages: positiveImages.size };
  if (mode === 'anom') { // fits on clean cells only; cracked cells are optional and only validate the threshold
    const cleanImages = new Set(negatives.map((i) => i.imageFile));
    if (negatives.length < 50) missing.push(`${negatives.length}/50 clean cells`);
    if (cleanImages.size < 2) missing.push(`${cleanImages.size}/2 images with clean cells`);
    if (cleanImages.size < 4) warnings.push(`Clean cells come from only ${cleanImages.size} image(s). Training splits by whole image and needs at least 20 clean cells to fit and 10 held out; with very few or very uneven images it can still stop with a data error.`);
    const autoNeg = negatives.filter((i) => i.source === 'auto-negative').length;
    if (autoNeg) warnings.push(`${autoNeg} clean cell(s) were auto-labelled (rest of a cracked module). If any of those modules hold an unmarked crack it is learned as normal; review them or turn off auto-negative in Cell Annotation.`);
    if (positives.length < 8 || positiveImages.size < 2) warnings.push('Fewer than 8 cracked cells in 2 images: the threshold will be set from clean cells alone (allowed false-alarm rate) and crack recall cannot be verified.');
    return { ready: missing.length === 0, missing, warnings, counts };
  }
  if (negatives.length < 100) warnings.push(`Only ${negatives.length} clean cells are labelled; precision at a 100:1 clean-to-cracked ratio may be unstable. Aim for at least 100 clean labels across multiple images.`);
  if (positives.length < 30) warnings.push(`Only ${positives.length} cracked cells are labelled; recall validation is weak below about 30 positives.`);
  if (positiveImages.size < 5) warnings.push(`Cracked labels currently cover ${positiveImages.size} source image(s); add more independent crack-positive images to validate generalization.`);
  if (images.size < 4) missing.push(`${images.size}/4 reviewed images`);
  if (positives.length < 8) missing.push(`${positives.length}/8 cracked cells`);
  if (positiveImages.size < 2) missing.push(`${positiveImages.size}/2 images with a cracked cell`);
  if (mode === 'cls' && negatives.length < 8) missing.push(`${negatives.length}/8 clean cells`);
  return { ready: missing.length === 0, missing, warnings, counts };
}

export function startClassifierTraining(mode: ClassifierMode) {
  const current = getClassifierTrainingStatus(mode);
  if (ACTIVE.includes(current.status)) throw new Error(`The ${MODE_NAME[mode]} is already training.`);
  if (!fs.existsSync(RUNTIME_MARKER)) throw new Error('Set up the YOLO training engine first (Training Centre -> Direct cell detection -> Set up).');
  const readiness = classifierReadiness(mode);
  if (!readiness.ready) throw new Error(`Not enough reviewed labels: ${readiness.missing.join(', ')}.`);

  const settings = getTrainingSettings(), s = settings.crackNet, an = settings.crackAnomaly;
  const cls = mode === 'cls';
  const jobId = crypto.randomUUID();
  const work = path.join(jobDir(mode), 'work');
  fs.mkdirSync(jobDir(mode), { recursive: true });
  fs.writeFileSync(stateFile(mode), JSON.stringify({ status: 'preparing', mode, jobId, progress: 0, message: 'Starting…', log: [], metrics: [], startedAt: new Date().toISOString() }));
  const common = ['--feedback-dir', FEEDBACK_DIR, '--work', work, '--state-file', stateFile(mode), '--cell-buffer', String(settings.crackClassifier.cellBuffer)];
  const args = mode === 'anom' ? [trainingWorkerPath('train_crack_anomaly.py'), ...common,
    '--activate', ANOMALY_ONNX, '--meta-activate', ANOMALY_META, '--backbone', an.backbone, '--imgsz', String(an.imgsz), '--memory-bank', String(an.memoryBankSize), '--top-k', String(an.topKPatches),
    '--val-fraction', String(an.valFraction), '--clean-quantile', String(an.cleanQuantile), '--target-precision', String(an.targetPrecision), '--clean-to-cracked-ratio', String(an.cleanToCrackedRatio),
    '--min-balanced-accuracy', String(an.minBalancedAccuracy), '--seed', String(an.seed)] : [trainingWorkerPath('train_crack_classifier.py'), '--mode', mode, ...common,
    '--activate', cls ? CLASSIFIER_ONNX : MODULE_ONNX, '--meta-activate', cls ? CLASSIFIER_META : MODULE_META,
    '--base-model', cls ? s.clsBase : s.detBase, '--epochs', String(cls ? s.clsEpochs : s.detEpochs), '--imgsz', String(cls ? s.clsImgsz : s.detImgsz),
    '--batch', String(cls ? s.clsBatch : s.detBatch), '--patience', String(cls ? s.clsPatience : s.detPatience), '--seed', String(s.seed),
    '--val-fraction', String(s.valFraction),
    '--target-recall', String(s.targetRecall), '--target-precision', String(s.targetPrecision), '--clean-to-cracked-ratio', String(s.realisticCleanToCrackedRatio), '--min-balanced-accuracy', String(s.minBalancedAccuracy), '--tile-overlap', String(s.detTileOverlap),
    '--edge-weight', String(settings.crackClassifier.edgeSampleWeight), '--hard-negative-weight', String(settings.crackClassifier.hardNegativeWeight),
    '--missed-crack-weight', String(settings.crackClassifier.missedCrackWeight)];
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
      if (ACTIVE.includes(state.status)) {
        let tail = ''; // the worker's own stderr (import errors etc.) is the only record when it died before writing state
        try { tail = fs.readFileSync(path.join(jobDir(mode), 'worker.log'), 'utf8').trim().split(/\r?\n/).slice(-12).join('\n'); } catch { /* no log */ }
        patchState(mode, { status: 'failed', message: `The training process exited unexpectedly (code ${code}). The previous model was kept.${tail ? ` Last worker output: ${tail.split('\n').slice(-1)[0]}` : ''}`, error: tail || undefined, log: [...(state.log ?? []), ...(tail ? tail.split('\n') : [])].slice(-500), finishedAt: new Date().toISOString() });
      }
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
export function installedClassifierModels(): { cls: Record<string, any> | null; det: Record<string, any> | null; anom: Record<string, any> | null; anomBundled: boolean; warning?: string; anomWarning?: string } {
  const read = (file: string, onnx: string) => { try { return fs.existsSync(onnx) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null; } catch { return null; } };
  const active = activeClassifierFiles(); // user-trained pair, else the bundled one
  return { cls: read(active.meta, active.onnx), det: read(MODULE_META, MODULE_ONNX), anom: (() => { const f = activeClassifierFiles('anomaly'), m = read(f.meta, f.onnx); return m ? { ...m, source: resolveAnomalyFiles()?.source ?? 'user' } : null; })(), anomBundled: bundledAnomalyFiles() !== null, warning: getCellClassifierWarning(), anomWarning: getCellClassifierWarning('anomaly') };
}

export const classifierLogText = (mode: ClassifierMode): string => (readState(mode).log ?? []).join('\n');

/** True while any of `modes` is being trained (a clear must never run underneath a live worker). */
export const isAnyClassifierTrainingActive = (modes: ClassifierMode[] = ['cls', 'det', 'anom']): boolean =>
  modes.some((mode) => children.has(mode) || ACTIVE.includes(getClassifierTrainingStatus(mode).status));

/** One snapshot for the Training Centre panel; shared by the status route and the clear route so both always return the same shape. */
export function classifierStatusSnapshot() {
  return { cls: getClassifierTrainingStatus('cls'), det: getClassifierTrainingStatus('det'), anom: getClassifierTrainingStatus('anom'), installed: installedClassifierModels(), readiness: { cls: classifierReadiness('cls'), det: classifierReadiness('det'), anom: classifierReadiness('anom') }, devices: getCrackInferenceDevices() };
}

/**
 * Clears the installed models of `scope` and everything their trainers generated (job state, logs, cropped/tiled datasets, Ultralytics runs).
 * 'archived' = A classifier + B detector (v1.4.14.4 behaviour); 'anomaly' = the anomaly detector.
 * Deliberately NOT touched: the reviewed cracked/clean labels and their source images (shared by every engine; cleared by "Clear crack dataset"),
 * the bundled read-only baseline models (clearing the anomaly detector only removes the user's trained pair; the bundled one then takes over), and the Python runtime / downloaded base weights. Refuses while a job of that scope runs.
 */
export function clearClassifierModelsAndData(scope: 'archived' | 'anomaly' = 'archived') {
  const modes: ClassifierMode[] = scope === 'anomaly' ? ['anom'] : ARCHIVED;
  if (isAnyClassifierTrainingActive(modes)) throw new Error('A model of this group is still training. Cancel it first, then clear.');
  // maxRetries/retryDelay: on Windows a model file can stay locked for a moment after its ONNX session is released.
  const remove = (target: string) => fs.rmSync(target, { recursive: true, force: true, maxRetries: 5, retryDelay: 150 });
  resetCellClassifierSession();
  resetModuleShadowSession(); // release the cached sessions first so the files are not held open
  const files = scope === 'anomaly' ? [ANOMALY_ONNX, ANOMALY_META] : [CLASSIFIER_ONNX, CLASSIFIER_META, MODULE_ONNX, MODULE_META];
  for (const file of files) { remove(file); remove(`${file}.tmp`); }
  for (const mode of modes) { remove(jobDir(mode)); reloaded.delete(mode); }
  return classifierStatusSnapshot();
}
