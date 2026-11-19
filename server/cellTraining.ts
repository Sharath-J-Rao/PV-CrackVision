/** App-managed YOLO11 cell training with durable job state and refresh-safe progress. */
import fs from 'node:fs';
import path from 'node:path';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import crypto from 'node:crypto';
import os from 'node:os';
import sharp from 'sharp';
import { maskUnannotatedAreas } from './partialAnnotationMask.ts';
import JSZip from 'jszip';
import { listCellAnnotations, cellAnnotationImagePath, clearCellAnnotations } from './cellAnnotationStore.ts';
import { getTrainingSettings, normalizeTrainingSettings, saveTrainingSettings } from './trainingSettings.ts';
import type { CellDetectorTrainingSettings } from '../src/types/trainingSettings.ts';
import { resetCellDetectorSession } from './cellDetector.ts';
import { clearSavedCellReviewData } from './reviewStore.ts';
import { buildPackageDependencyInstallArgs, buildPyPiWheelInstallArgs, buildTorchWheelInstallArgs, chooseTorchVariant, type TorchVariant } from './pythonInstallConfig.ts';
import { buildCellTrainingDatasetSummary } from './cellTrainingDatasetSummary.ts';
import { APP_ROOT, DATA_ROOT, CELL_YOLO_DIR, TRAINING_MODELS_DIR, VENV_DIR, EMBEDDED_PYTHON_DIR, MODEL_DATA_DIR, RUNTIME_MARKER, JOB_STATE_FILE, RUNTIME_DIR, legacyTrainingPythonCandidates } from './dataPaths.ts';
import { chooseValidationImageFiles } from './cellTrainingSplit.ts';
import { resolveExternalWorkerPath } from './externalWorkerPath.ts';

const execFileAsync = promisify(execFile);
const ROOT = APP_ROOT;
const ELECTRON_RESOURCES_PATH = (process as any).resourcesPath as string | undefined;
export const trainingWorkerPath = (file: string) => resolveExternalWorkerPath(`training/${file}`, ROOT, ELECTRON_RESOURCES_PATH);
const TRAIN_DIR = CELL_YOLO_DIR;
const DATASET_DIR = path.join(TRAIN_DIR, 'dataset');
const RUNS_DIR = path.join(TRAIN_DIR, 'runs');
const LOGS_DIR = path.join(TRAIN_DIR, 'logs');
const CHECKPOINT_DIR = TRAINING_MODELS_DIR;
const STABLE_BEST_CHECKPOINT = path.join(CHECKPOINT_DIR, 'cell-best.pt');
const EMBEDDED_DIR = EMBEDDED_PYTHON_DIR;
const EMBEDDED_PYTHON = path.join(EMBEDDED_DIR, 'python.exe');
const VENV_PYTHON = process.platform === 'win32' ? path.join(VENV_DIR, 'Scripts', 'python.exe') : path.join(VENV_DIR, 'bin', 'python');
let selectedPythonExecutable: string | undefined;
const preferredPythonExecutable = () => process.platform === 'win32' && fs.existsSync(EMBEDDED_PYTHON) ? EMBEDDED_PYTHON : VENV_PYTHON;
export const pythonExecutable = () => selectedPythonExecutable && fs.existsSync(selectedPythonExecutable) ? selectedPythonExecutable : preferredPythonExecutable();
const pythonExecutableCandidates = () => [...new Set([preferredPythonExecutable(), ...legacyTrainingPythonCandidates()])].filter((candidate) => fs.existsSync(candidate));
const MODEL_DIR = MODEL_DATA_DIR;
const ACTIVE_MODEL = path.join(MODEL_DIR, 'cell.onnx');
const CONTROL_FILE = `${JOB_STATE_FILE}.control`; // v1.4.10: the worker pauses while this file exists
const PREVIOUS_MODEL = path.join(MODEL_DIR, 'cell.previous.onnx');
// Job state stays outside the rebuildable YOLO dataset folder.

type TrainingStatus = 'not_ready' | 'setting_up' | 'ready' | 'training' | 'paused' | 'completed' | 'failed' | 'cancelled';
type State = {
  metrics?: Record<string, number>;
  onnxMetrics?: Record<string, number>;
  status: TrainingStatus;
  progress: number;
  message: string;
  log: string[];
  error?: string;
  startedAt?: string;
  finishedAt?: string;
  updatedAt?: string;
  jobId?: string;
  pid?: number;
  ownerPid?: number;
  epochMetrics?: Array<Record<string, string | number | null>>;
  logFile?: string;
  bestCheckpoint?: string;
  runDirectory?: string;
  checkpointSource?: string;
  errorTraceback?: string;
};

const defaultState = (): State => ({
  status: fs.existsSync(RUNTIME_MARKER) && fs.existsSync(pythonExecutable()) ? 'ready' : 'not_ready',
  progress: fs.existsSync(RUNTIME_MARKER) ? 100 : 0,
  message: fs.existsSync(RUNTIME_MARKER) ? 'YOLO training engine is ready.' : 'YOLO training engine has not been set up.',
  log: [],
});
function readSavedState(): State | null {
  try {
    const value = JSON.parse(fs.readFileSync(JOB_STATE_FILE, 'utf8')) as State;
    if (!value || !['not_ready', 'setting_up', 'ready', 'training', 'paused', 'completed', 'failed', 'cancelled'].includes(value.status)) return null;
    return { ...value, log: Array.isArray(value.log) ? value.log.slice(-300) : [], epochMetrics: Array.isArray(value.epochMetrics) ? value.epochMetrics.slice(-500) : [] };
  } catch { return null; }
}
function pidIsAlive(pid?: number): boolean {
  if (!pid || !Number.isInteger(pid) || pid < 1) return false;
  try { process.kill(pid, 0); return true; }
  catch (error) { return (error as NodeJS.ErrnoException).code === 'EPERM'; }
}
let state: State = readSavedState() ?? defaultState();
let activeChild: ReturnType<typeof spawn> | null = null;

/**
 * v1.4.12.6: the detached worker and this Node process both write JOB_STATE_FILE. Node used to re-save its own
 * (stale) in-memory copy, which could overwrite the worker's final `completed` record right after it exited and
 * produce a false "worker is no longer running" failure. Before every Node-side write, adopt the on-disk record
 * when it is newer than ours, so worker-written progress and the final status are never lost.
 */
function adoptNewerDiskState() {
  const saved = readSavedState();
  if (!saved) return;
  const diskTime = Date.parse(saved.updatedAt ?? '') || 0;
  const memoryTime = Date.parse(state.updatedAt ?? '') || 0;
  if (diskTime > memoryTime) state = saved;
}
/** Append a line to the per-job worker log file (best effort) so post-mortems never depend on state-only messages. */
function appendWorkerLog(line: string) {
  try { if (state.logFile) fs.appendFileSync(state.logFile, `[app] ${line}\n`); } catch { /* diagnostics must never break job handling */ }
}
function persistState() {
  state.updatedAt = new Date().toISOString();
  fs.mkdirSync(path.dirname(JOB_STATE_FILE), { recursive: true });
  const temp = `${JOB_STATE_FILE}.tmp`;
  fs.writeFileSync(temp, JSON.stringify(state, null, 2));
  try { fs.renameSync(temp, JOB_STATE_FILE); }
  catch {
    // Windows does not replace an existing destination on every filesystem.
    fs.rmSync(JOB_STATE_FILE, { force: true });
    fs.renameSync(temp, JOB_STATE_FILE);
  }
}
function setState(patch: Partial<State>) {
  adoptNewerDiskState(); // v1.4.12.6: apply the patch to the freshest record, not a stale copy
  state = { ...state, ...patch };
  persistState();
}
function pushLog(line: string) {
  adoptNewerDiskState(); // v1.4.12.6: never re-save a stale status over the worker's record
  state = { ...state, log: [...state.log.slice(-299), line.slice(0, 500)] };
  persistState();
}
/** v1.4.11.7: work out WHY a worker died from its saved log (traceback, OOM, paging-file, native crash) instead of a generic message. */
function diagnoseWorkerDeath(): string {
  const hints: string[] = [];
  try {
    if (state.logFile && fs.existsSync(state.logFile)) {
      const size = fs.statSync(state.logFile).size;
      const fd = fs.openSync(state.logFile, 'r');
      const len = Math.min(size, 64 * 1024), buf = Buffer.alloc(len);
      fs.readSync(fd, buf, 0, len, size - len); fs.closeSync(fd);
      const tail = buf.toString('utf8');
      const lines = tail.split(/\r?\n/).filter((l) => l.trim());
      const tb = tail.lastIndexOf('Traceback (most recent call last)');
      const err = [...lines].reverse().find((l) => /^\s*(\w+\.)*\w*(Error|Exception|Interrupt|Exit)\b.*:|TRAINING_ERROR|CUDA out of memory|MemoryError|Fatal Python error|access violation|1455|paging file|DefaultCPUAllocator/i.test(l));
      if (/out of memory|MemoryError|DefaultCPUAllocator|not enough memory/i.test(tail)) hints.push('Out of memory: lower Batch size and/or Image size in Settings (try batch 2, imgsz 512) and close other apps.');
      if (/paging file|error code 1455|shm|shared memory/i.test(tail)) hints.push('Windows paging file too small: increase virtual memory, or lower Batch size.');
      if (/Fatal Python error|access violation|segmentation fault|0xC0000005/i.test(tail)) hints.push('Native crash inside PyTorch/driver: update GPU drivers or set PV_YOLO_DEVICE=cpu.');
      if (err) hints.push(`Last error in log: ${err.trim().slice(0, 300)}`);
      else if (tb >= 0) hints.push(`Traceback in log: ${tail.slice(tb).split(/\r?\n/).filter((l) => l.trim()).slice(-3).join(' | ').slice(0, 400)}`);
      else if (lines.length) hints.push(`No error was written. Last log line: ${lines[lines.length - 1].trim().slice(0, 200)}`);
      else hints.push('The worker log is empty.');
    } else hints.push('No worker log file was found.');
  } catch (e) { hints.push(`Could not read the worker log: ${e instanceof Error ? e.message : String(e)}`); }
  if (!hints.some((h) => /Out of memory|paging|Native crash/.test(h))) hints.push('Likely the process was killed by the OS (memory pressure/antivirus) or the app was closed; check Task Manager memory and the worker log.');
  return hints.join(' ');
}

function refreshStateFromDisk() {
  const saved = readSavedState();
  if (saved) {
    const diskTime = Date.parse(saved.updatedAt ?? '') || 0;
    const memoryTime = Date.parse(state.updatedAt ?? '') || 0;
    const sameJobFinished = Boolean(saved.jobId) && saved.jobId === state.jobId && ['completed', 'failed', 'cancelled'].includes(saved.status);
    // v1.4.12.6: a finished record of the running job always wins, whatever the timestamps say.
    if (diskTime >= memoryTime || sameJobFinished) state = saved;
  }
  if ((state.status === 'training' || state.status === 'paused') && !pidIsAlive(state.pid)) {
    setState({
      status: 'failed',
      message: 'The training worker stopped before recording a successful completion.',
      error: `The training worker is no longer running. ${diagnoseWorkerDeath()} The saved log and last progress are retained; the previously active model was not replaced.`,
      finishedAt: new Date().toISOString(),
    });
  } else if (state.status === 'setting_up' && state.ownerPid && state.ownerPid !== process.pid) {
    setState({
      status: 'failed',
      message: 'Training runtime setup was interrupted by an application restart.',
      error: 'The setup state was recovered from disk. Start runtime setup again; existing downloaded packages are retained.',
      finishedAt: new Date().toISOString(),
    });
  }
}

// Recover a worker that is still running even when the browser or Node server was restarted.
refreshStateFromDisk();

const run = async (command: string, args: string[], timeout = 30 * 60 * 1000, env: NodeJS.ProcessEnv = process.env) => {
  const { stdout, stderr } = await execFileAsync(command, args, { cwd: DATA_ROOT, timeout, windowsHide: true, maxBuffer: 12 * 1024 * 1024, env });
  if (stdout.trim()) pushLog(stdout.trim().slice(-1200));
  if (stderr.trim()) pushLog(stderr.trim().slice(-1200));
  return [stdout, stderr].filter(Boolean).join('\n').trim();
};

async function findPython(): Promise<{ command: string; prefix: string[]; version: string }> {
  const candidates = process.platform === 'win32' ? [{ command: 'py', prefix: ['-3'] }, { command: 'python', prefix: [] }] : [{ command: 'python3', prefix: [] }, { command: 'python', prefix: [] }];
  for (const candidate of candidates) {
    try {
      const version = await run(candidate.command, [...candidate.prefix, '--version'], 15000);
      const match = version.match(/Python\s+(\d+)\.(\d+)/i);
      if (match && Number(match[1]) === 3 && Number(match[2]) >= 10 && Number(match[2]) <= 12) return { ...candidate, version };
    } catch { /* try next interpreter */ }
  }
  throw new Error('Python 3.10–3.12 was not found. Install Python once, enable the Python launcher/PATH, restart PV CrackVision, then use Set up training engine. YOLO training itself is launched and managed inside this app.');
}
function venvExists() { return pythonExecutableCandidates().length > 0; }

async function installPortablePythonWindows() {
  if (process.platform !== 'win32') throw new Error('Automatic embedded Python setup is currently available on Windows. Install Python 3.10–3.12 on macOS/Linux and retry.');
  fs.mkdirSync(EMBEDDED_DIR, { recursive: true });
  setState({ message: 'Downloading the app-managed Python runtime…', progress: 5 });
  const pythonUrl = 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip';
  const pythonResponse = await fetch(pythonUrl, { signal: AbortSignal.timeout(5 * 60 * 1000) });
  if (!pythonResponse.ok) throw new Error(`Could not download the embedded Python runtime (HTTP ${pythonResponse.status}).`);
  const archive = await JSZip.loadAsync(Buffer.from(await pythonResponse.arrayBuffer()));
  for (const [name, entry] of Object.entries(archive.files)) {
    if (entry.dir) { fs.mkdirSync(path.join(EMBEDDED_DIR, name), { recursive: true }); continue; }
    const destination = path.join(EMBEDDED_DIR, name); fs.mkdirSync(path.dirname(destination), { recursive: true }); fs.writeFileSync(destination, await entry.async('nodebuffer'));
  }
  const pth = path.join(EMBEDDED_DIR, 'python312._pth');
  if (!fs.existsSync(pth)) throw new Error('Embedded Python archive did not contain python312._pth.');
  let config = fs.readFileSync(pth, 'utf8').replace(/^#import site/m, 'import site');
  if (!config.includes('Lib\\site-packages')) config += '\nLib\\site-packages\n';
  fs.writeFileSync(pth, config);
  setState({ message: 'Bootstrapping pip in the app-managed Python runtime…', progress: 10 });
  const pipResponse = await fetch('https://bootstrap.pypa.io/get-pip.py', { signal: AbortSignal.timeout(2 * 60 * 1000) });
  if (!pipResponse.ok) throw new Error(`Could not download pip bootstrap (HTTP ${pipResponse.status}).`);
  const getPip = path.join(EMBEDDED_DIR, 'get-pip.py'); fs.writeFileSync(getPip, Buffer.from(await pipResponse.arrayBuffer()));
  await run(EMBEDDED_PYTHON, [getPip, '--disable-pip-version-check'], 10 * 60 * 1000);
}

/** Which PyTorch wheel this machine should use (GPU when an NVIDIA driver is present). */
async function detectTorchVariant(): Promise<TorchVariant> {
  let smi: string | undefined;
  try { smi = await run('nvidia-smi', [], 15000); } catch { smi = undefined; } // no NVIDIA driver -> CPU wheels
  return chooseTorchVariant(process.env.PV_TORCH_VARIANT, smi);
}
/** True when the installed torch build can actually see a CUDA device. */
async function torchHasCuda(): Promise<boolean> {
  try {
    const out = await run(pythonExecutable(), ['-c', 'import torch; print("CUDA_OK" if torch.cuda.is_available() else "CUDA_NO")'], 120000);
    return out.includes('CUDA_OK');
  } catch { return false; }
}
/** Replace a CPU-only torch with the GPU build. Non-fatal: training still works on CPU if this fails. */
async function upgradeTorchForGpu(env: NodeJS.ProcessEnv, dependencyHelper: string): Promise<void> {
  const variant = await detectTorchVariant();
  if (variant === 'cpu' || await torchHasCuda()) return;
  setState({ progress: 30, message: `NVIDIA GPU detected - installing PyTorch ${variant} (GPU) wheels…` });
  try {
    await run(pythonExecutable(), buildTorchWheelInstallArgs(variant, true), 60 * 60 * 1000, env);
    await run(pythonExecutable(), buildPackageDependencyInstallArgs(dependencyHelper, ['torch', 'torchvision']), 60 * 60 * 1000, env);
    pushLog(`Installed GPU PyTorch (${variant}); CUDA available: ${await torchHasCuda()}.`);
  } catch (error) {
    pushLog(`GPU PyTorch install failed (${error instanceof Error ? error.message : String(error)}); training will use the CPU build.`);
  }
}

async function probeTrainingPackages(): Promise<{ torch: boolean; torchvision: boolean; ultralytics: boolean; onnxruntime: boolean }> {
  const script = "import importlib.util,json; print(json.dumps({'torch': importlib.util.find_spec('torch') is not None, 'torchvision': importlib.util.find_spec('torchvision') is not None, 'ultralytics': importlib.util.find_spec('ultralytics') is not None, 'onnxruntime': importlib.util.find_spec('onnxruntime') is not None}))";
  const candidates = pythonExecutableCandidates();
  let lastError: unknown;
  for (const candidate of candidates) {
    try {
      const output = await run(candidate, ['-c', script], 30000);
      const line = output.split(/\r?\n/).reverse().find((part) => part.trim().startsWith('{'));
      if (!line) throw new Error(`Could not read the YOLO runtime package status from ${candidate}.`);
      selectedPythonExecutable = candidate;
      return JSON.parse(line) as { torch: boolean; torchvision: boolean; ultralytics: boolean; onnxruntime: boolean };
    } catch (error) { lastError = error; }
  }
  throw new Error(`No reusable shared or legacy Python runtime could be started. ${lastError instanceof Error ? lastError.message : String(lastError ?? 'No Python runtime was found.')}`);
}

function newestBestCheckpoint(): string | undefined {
  if (fs.existsSync(STABLE_BEST_CHECKPOINT)) return STABLE_BEST_CHECKPOINT;
  const candidates: Array<{ file: string; time: number }> = [];
  // The current worker only writes state.bestCheckpoint after the validation
  // gate has passed. Older releases did not persist that field, so their run
  // folders are accepted only when their saved validation_metrics.json passes
  // the same gate; rejected candidates are never used for continued training.
  if (state.bestCheckpoint && fs.existsSync(state.bestCheckpoint)) {
    candidates.push({ file: state.bestCheckpoint, time: fs.statSync(state.bestCheckpoint).mtimeMs });
  }
  if (fs.existsSync(RUNS_DIR)) {
    for (const entry of fs.readdirSync(RUNS_DIR, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const file = path.join(RUNS_DIR, entry.name, 'weights', 'best.pt');
      const metricsFile = path.join(RUNS_DIR, entry.name, 'validation_metrics.json');
      if (!fs.existsSync(file) || !fs.existsSync(metricsFile)) continue;
      try {
        const metrics = JSON.parse(fs.readFileSync(metricsFile, 'utf8')) as Record<string, number>;
        const map50 = metrics['metrics/mAP50(B)'] ?? metrics['metrics/mAP50'] ?? 0;
        const precision = metrics['metrics/precision(B)'] ?? metrics['metrics/precision'] ?? 0;
        const recall = metrics['metrics/recall(B)'] ?? metrics['metrics/recall'] ?? 0;
        if (map50 >= 0.05 && (precision > 0 || recall > 0)) candidates.push({ file, time: fs.statSync(file).mtimeMs });
      } catch { /* malformed/missing validation files are not trusted checkpoints */ }
    }
  }
  candidates.sort((a, b) => b.time - a.time);
  return candidates[0]?.file;
}

/** Read the reusable YOLO cell checkpoint for optional inclusion in a portable data backup. */
export function getCellTrainingCheckpointBuffer(): Buffer | null {
  try { return fs.existsSync(STABLE_BEST_CHECKPOINT) ? fs.readFileSync(STABLE_BEST_CHECKPOINT) : null; }
  catch { return null; }
}

export function getCellTrainingDatasetAudit(): string | undefined {
  const auditPath = path.join(DATASET_DIR, 'dataset_audit.json');
  return fs.existsSync(auditPath) ? fs.readFileSync(auditPath, 'utf8') : undefined;
}

export function getCellTrainingLogPath(): string | undefined {
  refreshStateFromDisk();
  return state.logFile && fs.existsSync(state.logFile) ? state.logFile : undefined;
}

export function getCellTrainingMetricsCsv(): string {
  refreshStateFromDisk();
  const columns = ['epoch', 'timestamp', 'precision', 'recall', 'map50', 'map50_95', 'box_loss', 'cls_loss', 'dfl_loss', 'val_box_loss', 'val_cls_loss', 'val_dfl_loss'];
  const quote = (value: unknown) => `"${String(value ?? '').replace(/"/g, '""')}"`;
  return [columns.join(','), ...(state.epochMetrics ?? []).map((row) => columns.map((column) => quote(row[column])).join(','))].join('\r\n') + '\r\n';
}

type TrainingSourceRecord = { partial?: boolean; missedBoxes?: Array<{ left: number; top: number; width: number; height: number }>; id: string; fileName: string; imageFile: string; boxes: Array<{ left: number; top: number; width: number; height: number }>; imagePath: string; sampleType: 'positive' | 'negative' };

/** Build YOLO localization data only from complete, explicitly reviewed cell-box images. Crack labels are a separate dataset and must never become partial YOLO box labels. */
function getTrainingSourceRecords(): TrainingSourceRecord[] {
  const byImage = new Map<string, TrainingSourceRecord>();
  const addBoxes = (key: string, fileName: string, imageFile: string, imagePath: string, boxes: Array<{ left: number; top: number; width: number; height: number }>, sampleType: 'positive' | 'negative' = 'positive', missed: Array<{ left: number; top: number; width: number; height: number }> = [], partial = false) => {
    if (!fs.existsSync(imagePath) || (!boxes.length && sampleType !== 'negative')) return;
    const current: TrainingSourceRecord = byImage.get(key) ?? { id: key, fileName, imageFile, boxes: [], imagePath, sampleType, partial };
    current.partial = current.partial === true && partial; // a merged image counts as partial only if every source says so
    for (const box of boxes) {
      const cx = box.left + box.width / 2, cy = box.top + box.height / 2;
      const duplicate = current.boxes.some((prior) => {
        const px = prior.left + prior.width / 2, py = prior.top + prior.height / 2;
        const tolerance = Math.max(4, Math.min(prior.width, prior.height, box.width, box.height) * 0.18);
        return Math.hypot(px - cx, py - cy) <= tolerance;
      });
      if (!duplicate) current.boxes.push({ left: box.left, top: box.top, width: box.width, height: box.height });
    }
    if (current.boxes.length) current.sampleType = 'positive';
    else if (sampleType === 'negative') current.sampleType = 'negative';
    if (missed.length) current.missedBoxes = [...(current.missedBoxes ?? []), ...missed];
    byImage.set(key, current);
  };

  for (const record of listCellAnnotations()) {
    const imagePath = cellAnnotationImagePath(record);
    addBoxes(record.imageFile, record.fileName || record.imageFile, record.imageFile, imagePath, record.boxes ?? [], record.sampleType === 'negative' ? 'negative' : 'positive', record.missedBoxes ?? [], record.partial === true);
  }
  // Do not import crack-classifier feedback here. Those labels are intentionally
  // sparse (only cells selected by an operator) and all unboxed cells would be
  // treated as background by YOLO, poisoning localization training.
  return [...byImage.values()];
}

export function getCellTrainingStatus() {
  refreshStateFromDisk();
  const records = getTrainingSourceRecords();
  const positiveImages = records.filter((record) => record.boxes.length > 0).length;
  const negativeSamples = records.filter((record) => record.sampleType === 'negative' && record.boxes.length === 0).length;
  const boxes = records.reduce((n, r) => n + r.boxes.length, 0);
  return { ...state, runtimeReady: venvExists() && fs.existsSync(RUNTIME_MARKER) && state.status !== 'setting_up', annotatedImages: positiveImages, positiveImages, negativeSamples, totalTrainingImages: records.length, annotatedCells: boxes, minimumImages: 2, trainingSources: records.map((record) => ({ fileName: record.fileName, imageFile: record.imageFile, boxes: record.boxes.length, sampleType: record.sampleType })), activeModelExists: fs.existsSync(ACTIVE_MODEL), activeModel: path.join('data', 'models', 'cell.onnx'), previousModelExists: fs.existsSync(PREVIOUS_MODEL), onnxRuntimeReady: fs.existsSync(path.join(ROOT, 'node_modules', 'onnxruntime-node', 'package.json')), pageRefreshSafe: true, logDownloadUrl: state.logFile && fs.existsSync(state.logFile) ? '/api/cell-training/log' : undefined, metricsDownloadUrl: (state.epochMetrics?.length ?? 0) > 0 ? '/api/cell-training/metrics.csv' : undefined, datasetAuditUrl: fs.existsSync(path.join(DATASET_DIR, 'dataset_audit.json')) ? '/api/cell-training/dataset-audit' : undefined, checkpointReusable: !!newestBestCheckpoint(), reusableCheckpoint: newestBestCheckpoint() ? path.relative(DATA_ROOT, newestBestCheckpoint()!).replace(/\\/g, '/') : undefined };
}

export async function setupCellTrainingEngine() {
  refreshStateFromDisk();
  if (state.status === 'setting_up' || state.status === 'training' || state.status === 'paused') throw new Error('A cell training operation is already running (or paused; resume or abort it first).');
  const jobId = crypto.randomUUID();
  setState({ status: 'setting_up', progress: 1, message: 'Checking the app-managed Python and YOLO runtime…', error: undefined, errorTraceback: undefined, startedAt: new Date().toISOString(), finishedAt: undefined, jobId, pid: undefined, ownerPid: process.pid });
  pushLog('Verifying the existing Python, PyTorch, Torchvision, and Ultralytics installation.');
  try {
    if (!venvExists()) {
      try {
        const python = await findPython();
        pushLog(`Using ${python.version}`);
        await run(python.command, [...python.prefix, '-m', 'venv', VENV_DIR], 5 * 60 * 1000);
      } catch (pythonError) {
        if (process.platform !== 'win32') throw pythonError;
        pushLog('No compatible system Python found; installing the private embedded Python runtime.');
        await installPortablePythonWindows();
      }
    }

    let packages = await probeTrainingPackages();
    if (packages.torch && packages.torchvision && packages.ultralytics) {
      // A runtime created by an older release holds CPU-only torch; switch it to the GPU build
      // when an NVIDIA GPU is present (no-op on CPU-only machines or if CUDA already works).
      const gpuEnv = { ...process.env };
      delete gpuEnv.PIP_INDEX_URL; delete gpuEnv.PIP_EXTRA_INDEX_URL; delete gpuEnv.PIP_FIND_LINKS;
      gpuEnv.PIP_CONFIG_FILE = os.devNull;
      await upgradeTorchForGpu(gpuEnv, trainingWorkerPath('install_package_dependencies.py'));
      // Validate the exported ONNX model with the CPU ONNX runtime (device-independent check) before activation.
      await run(pythonExecutable(), ['-m', 'pip', 'check'], 60000);
      const version = await run(pythonExecutable(), ['-c', 'import torch, torchvision, ultralytics; print(ultralytics.__version__)'], 60000);
      if (!packages.onnxruntime) {
        setState({ progress: 82, message: 'Installing the CPU ONNX validation runtime…' });
        const cleanEnv = { ...process.env };
        delete cleanEnv.PIP_INDEX_URL; delete cleanEnv.PIP_EXTRA_INDEX_URL; delete cleanEnv.PIP_FIND_LINKS;
        cleanEnv.PIP_CONFIG_FILE = os.devNull;
        await run(pythonExecutable(), ['-m', 'pip', 'install', 'onnxruntime'], 60 * 60 * 1000, cleanEnv);
        packages = await probeTrainingPackages();
      }
      if (!packages.onnxruntime) throw new Error('Python onnxruntime could not be installed; exported ONNX models cannot be validated safely.');
      await run(pythonExecutable(), ['-m', 'pip', 'check'], 60000);
      fs.writeFileSync(RUNTIME_MARKER, `Ultralytics ${version}\nONNX Runtime ready\n${new Date().toISOString()}\n`);
      setState({ status: 'ready', progress: 100, message: `Existing YOLO runtime verified (Ultralytics ${version}; ONNX validation runtime ready).`, finishedAt: new Date().toISOString() });
      return getCellTrainingStatus();
    }

    // Do not upgrade pip or reinstall dependencies on every click. Install only
    // package groups that are actually missing from the local app venv.
    try { await run(pythonExecutable(), ['-m', 'pip', '--version'], 15000); }
    catch { await run(pythonExecutable(), ['-m', 'ensurepip', '--upgrade'], 120000); }
    const pipEnv = { ...process.env };
    // Keep user/system pip configuration from silently adding a second index
    // or local find-links source to the deliberately selected package index.
    delete pipEnv.PIP_INDEX_URL;
    delete pipEnv.PIP_EXTRA_INDEX_URL;
    delete pipEnv.PIP_FIND_LINKS;
    pipEnv.PIP_CONFIG_FILE = os.devNull;
    const dependencyHelper = trainingWorkerPath('install_package_dependencies.py');
    if (!packages.torch || !packages.torchvision) {
      const torchVariant = await detectTorchVariant();
      setState({ progress: 22, message: `Installing official PyTorch wheels (${torchVariant === 'cpu' ? 'CPU' : `GPU ${torchVariant}`})…` });
      // Binary wheels only, from the variant's PyTorch index. No dependency/build-backend
      // lookup is performed against the PyTorch-only index.
      await run(pythonExecutable(), buildTorchWheelInstallArgs(torchVariant), 60 * 60 * 1000, pipEnv);
      setState({ progress: 45, message: 'Resolving PyTorch dependencies from PyPI…' });
      // Read Requires-Dist from the installed CPU wheels. The helper installs
      // those dependencies from PyPI and pins torch/torchvision to the exact
      // installed versions so the CPU wheels cannot be replaced.
      await run(pythonExecutable(), buildPackageDependencyInstallArgs(dependencyHelper, ['torch', 'torchvision']), 60 * 60 * 1000, pipEnv);
    }
    if (!packages.ultralytics) {
      setState({ progress: 68, message: 'Installing Ultralytics YOLO11 from PyPI…' });
      // Install the Ultralytics wheel without dependencies, then resolve its
      // metadata-declared dependencies separately while preserving CPU torch.
      await run(pythonExecutable(), buildPyPiWheelInstallArgs('ultralytics'), 60 * 60 * 1000, pipEnv);
      await run(pythonExecutable(), buildPackageDependencyInstallArgs(dependencyHelper, ['ultralytics']), 60 * 60 * 1000, pipEnv);
    }
    packages = await probeTrainingPackages();
    if (!packages.torch || !packages.torchvision || !packages.ultralytics) throw new Error(`Training runtime verification failed. Installed packages: ${JSON.stringify(packages)}`);
    if (!packages.onnxruntime) {
      setState({ progress: 84, message: 'Installing the CPU ONNX validation runtime…' });
      await run(pythonExecutable(), ['-m', 'pip', 'install', 'onnxruntime'], 60 * 60 * 1000, pipEnv);
      packages = await probeTrainingPackages();
    }
    if (!packages.onnxruntime) throw new Error(`Python ONNX Runtime verification failed. Installed packages: ${JSON.stringify(packages)}`);
    // Catch missing or incompatible transitive dependencies before marking setup ready.
    await run(pythonExecutable(), ['-m', 'pip', 'check'], 60000);
    const version = await run(pythonExecutable(), ['-c', 'import torch, torchvision, ultralytics; print(ultralytics.__version__)'], 60000);
    fs.writeFileSync(RUNTIME_MARKER, `Ultralytics ${version}\n${new Date().toISOString()}\n`);
    setState({ status: 'ready', progress: 100, message: `YOLO training engine ready (Ultralytics ${version}). Runtime verified; future setup checks will reuse these packages.`, finishedAt: new Date().toISOString() });
    return getCellTrainingStatus();
  } catch (error) {
    fs.rmSync(RUNTIME_MARKER, { force: true });
    const details = error as { stderr?: string };
    setState({ status: 'failed', message: 'Could not prepare the YOLO training engine.', error: `${error instanceof Error ? error.message : String(error)}${details.stderr ? `\n${details.stderr.slice(-2500)}` : ''}`, finishedAt: new Date().toISOString() });
    throw error;
  }
}

/** Exported for tests. Builds the YOLO dataset; images with operator-added (missed) cells get extra TRAIN copies. */
export async function createDataset() {
  const missedRepeat = getTrainingSettings().cellDetector.missedCellRepeat;
  const partialMargin = getTrainingSettings().cellDetector.partialMaskMargin;
  let missedCellImages = 0, missedCells = 0, missedCopies = 0, partialMaskedImages = 0;
  const sourceRecords = getTrainingSourceRecords();
  type CanonicalTrainingRecord = { partial: boolean; missedCount: number; fileName: string; imageFile: string; imageData: Buffer; width: number; height: number; boxes: TrainingSourceRecord['boxes']; sampleType: 'positive' | 'negative' };
  const canonical = new Map<string, CanonicalTrainingRecord>();
  for (const source of sourceRecords) {
    const normalized = await sharp(fs.readFileSync(source.imagePath)).rotate().png().toBuffer({ resolveWithObject: true });
    const key = crypto.createHash('sha256').update(normalized.data).digest('hex');
    const current: CanonicalTrainingRecord = canonical.get(key) ?? { partial: source.partial === true, missedCount: 0, fileName: source.fileName, imageFile: source.imageFile, imageData: normalized.data, width: normalized.info.width, height: normalized.info.height, boxes: [], sampleType: source.sampleType };
    for (const box of source.boxes) {
      const cx = box.left + box.width / 2, cy = box.top + box.height / 2;
      const duplicate = current.boxes.some((prior) => {
        const px = prior.left + prior.width / 2, py = prior.top + prior.height / 2;
        const tolerance = Math.max(4, Math.min(prior.width, prior.height, box.width, box.height) * 0.18);
        return Math.hypot(px - cx, py - cy) <= tolerance;
      });
      if (!duplicate) current.boxes.push(box);
    }
    current.partial = current.partial && source.partial === true;
    current.missedCount += source.missedBoxes?.length ?? 0;
    if (current.boxes.length) current.sampleType = 'positive';
    canonical.set(key, current);
  }
  const records = [...canonical.values()];
  const positiveRecords = records.filter((record) => record.boxes.length > 0);
  const negativeRecords = records.filter((record) => record.boxes.length === 0 && record.sampleType === 'negative');
  if (positiveRecords.length < 2) throw new Error(`At least 2 distinct positive cell-annotated images are required for an independent train/validation split; currently ${positiveRecords.length}. Add background-only negatives separately; they do not replace positive cell annotations.`);
  fs.rmSync(DATASET_DIR, { recursive: true, force: true });
  for (const split of ['train', 'val']) { fs.mkdirSync(path.join(DATASET_DIR, 'images', split), { recursive: true }); fs.mkdirSync(path.join(DATASET_DIR, 'labels', split), { recursive: true }); }
  const orderedPositive = [...positiveRecords].sort((a, b) => a.imageFile.localeCompare(b.imageFile));
  const validationImageFiles = chooseValidationImageFiles(orderedPositive.map((record) => ({ imageFile: record.imageFile, boxes: record.boxes })));
  // Hold out a share of background-only images too, when there are enough to
  // keep at least one background sample on both sides of the split.
  const orderedNegative = [...negativeRecords].sort((a, b) => crypto.createHash('sha256').update(a.imageFile).digest('hex').localeCompare(crypto.createHash('sha256').update(b.imageFile).digest('hex')));
  const negativeValidationCount = orderedNegative.length >= 2 ? Math.max(1, Math.min(orderedNegative.length - 1, Math.ceil(orderedNegative.length * 0.2))) : 0;
  for (const record of orderedNegative.slice(0, negativeValidationCount)) validationImageFiles.add(record.imageFile);
  const ordered = [...records].sort((a, b) => a.imageFile.localeCompare(b.imageFile));
  let totalBoxes = 0, trainBoxes = 0, valBoxes = 0;
  let trainNegativeImages = 0, validationNegativeImages = 0;
  const splitAudit: Array<{ image: string; imageFile: string; sha256: string; width: number; height: number; split: string; boxes: number; sampleType: string }> = [];
  for (let i = 0; i < ordered.length; i++) {
    const record = ordered[i];
    const split = validationImageFiles.has(record.imageFile) ? 'val' : 'train';
    const w = record.width, h = record.height;
    if (!w || !h) continue;
    const base = `pv_${String(i).padStart(5, '0')}`;
    // v1.4.12.8: images flagged "partially annotated" keep only the marked cells (plus a small margin); the rest is painted
    // neutral grey so YOLO does not learn that real but unmarked cells are background. Applies to train and val alike.
    const isPartial = record.partial && record.boxes.length > 0;
    const trainingImage = isPartial
      ? await maskUnannotatedAreas(record.imageData, record.boxes, partialMargin)
      : await sharp(record.imageData).jpeg({ quality: 95 }).toBuffer();
    if (isPartial) partialMaskedImages++;
    fs.writeFileSync(path.join(DATASET_DIR, 'images', split, `${base}.jpg`), trainingImage);
    const unique = new Map<string, string>();
    for (const b of record.boxes) {
      const values = [b.left, b.top, b.width, b.height].map(Number);
      if (!values.every(Number.isFinite) || b.width <= 1 || b.height <= 1 || b.left < 0 || b.top < 0 || b.left + b.width > w + 2 || b.top + b.height > h + 2) {
        throw new Error(`Invalid cell box in ${record.fileName || record.imageFile}. Reopen this image in Advanced Annotations and correct boxes that are outside the image or have zero/negative dimensions.`);
      }
      const left = Math.max(0, Math.min(w, b.left)), top = Math.max(0, Math.min(h, b.top));
      const width = Math.max(1, Math.min(w - left, b.width)), height = Math.max(1, Math.min(h - top, b.height));
      const line = `0 ${((left + width / 2) / w).toFixed(6)} ${((top + height / 2) / h).toFixed(6)} ${(width / w).toFixed(6)} ${(height / h).toFixed(6)}`;
      unique.set(line, line);
    }
    // YOLO convention: an existing empty label file is a true background image.
    fs.writeFileSync(path.join(DATASET_DIR, 'labels', split, `${base}.txt`), unique.size ? [...unique.values()].join('\n') + '\n' : '');
    // v1.4.11.5: hard-example mining. Cells the model missed are repeated in TRAIN only (validation stays untouched).
    if (record.missedCount > 0 && unique.size) {
      missedCellImages++; missedCells += record.missedCount;
      for (let copy = 1; split === 'train' && copy <= missedRepeat; copy++) {
        fs.writeFileSync(path.join(DATASET_DIR, 'images', split, `${base}_m${copy}.jpg`), trainingImage);
        fs.writeFileSync(path.join(DATASET_DIR, 'labels', split, `${base}_m${copy}.txt`), [...unique.values()].join('\n') + '\n');
        missedCopies++;
      }
    }
    totalBoxes += unique.size;
    if (split === 'train') trainBoxes += unique.size; else valBoxes += unique.size;
    if (!unique.size) { if (split === 'train') trainNegativeImages++; else validationNegativeImages++; }
    splitAudit.push({ image: record.fileName || record.imageFile, imageFile: record.imageFile, sha256: crypto.createHash('sha256').update(record.imageData).digest('hex'), width: w, height: h, split, boxes: unique.size, sampleType: unique.size ? 'positive' : 'negative' });
  }
  const trainImageCount = splitAudit.filter((x) => x.split === 'train').length;
  const validationImageCount = splitAudit.filter((x) => x.split === 'val').length;
  if (!trainImageCount || !validationImageCount) throw new Error('Could not create non-empty train and validation image sets.');
  if (trainBoxes < 1 || valBoxes < 1) throw new Error(`The independent split needs at least one positive cell box in both sets; found ${trainBoxes} train and ${valBoxes} validation boxes.`);
  const audit = { partialMaskedImages, partialMaskMargin: partialMargin, missedCellImages, missedCells, missedCellExtraTrainCopies: missedCopies, missedCellRepeat: missedRepeat, createdAt: new Date().toISOString(), totalImages: splitAudit.length, trainImages: trainImageCount, validationImages: validationImageCount, trainBoxes, validationBoxes: valBoxes, totalBoxes, negativeImages: negativeRecords.length, trainNegativeImages, validationNegativeImages, split: splitAudit, sources: ['Explicit full-image YOLO cell-box annotations only', 'Explicit background-only negative samples'], excludedSources: ['Crack-classifier positive/negative feedback (sparse cell labels; never used as YOLO localization labels)'], warning: 'YOLO treats every unboxed visible cell as background. A negative sample must contain no visible PV cells to detect; do not upload partially annotated PV images as background.' };
  fs.writeFileSync(path.join(DATASET_DIR, 'dataset_audit.json'), JSON.stringify(audit, null, 2));
  const yaml = `path: ${JSON.stringify(DATASET_DIR.replace(/\\/g, '/'))}\ntrain: images/train\nval: images/val\nnames:\n  0: pv_cell\n`;
  const yamlPath = path.join(DATASET_DIR, 'dataset.yaml');
  fs.writeFileSync(yamlPath, yaml);
  return Object.assign(buildCellTrainingDatasetSummary({ yamlPath, images: records.length, boxes: totalBoxes, trainImages: trainImageCount, validationImages: validationImageCount, trainBoxes, validationBoxes: valBoxes }), { missedCellImages, missedCells, missedCopies });
}

export async function startCellYoloTraining(options: Partial<CellDetectorTrainingSettings> = {}) {
  refreshStateFromDisk();
  if (state.status === 'setting_up' || state.status === 'training' || state.status === 'paused') throw new Error('A cell training operation is already running (or paused; resume or abort it first).');
  if (!venvExists() || !fs.existsSync(RUNTIME_MARKER)) throw new Error('Set up the YOLO training engine inside the app first.');
  const settings = normalizeTrainingSettings({ cellDetector: { ...getTrainingSettings().cellDetector, ...options } }).cellDetector;
  const { epochs, imageSize, batch } = { epochs: settings.epochs, imageSize: settings.imageSize, batch: settings.batch };
  const dataset = await createDataset();
  fs.mkdirSync(RUNS_DIR, { recursive: true });
  fs.mkdirSync(LOGS_DIR, { recursive: true });
  const script = trainingWorkerPath('train_cell_yolo.py');
  const jobId = crypto.randomUUID();
  const initialWeights = newestBestCheckpoint();
  const runName = `pv_cell_${new Date().toISOString().replace(/[-:TZ.]/g, '').slice(0, 14)}_${jobId.slice(0, 6)}`;
  const logFile = path.join(LOGS_DIR, `cell-yolo-${jobId}.log`);
  setState({ status: 'training', progress: 1, message: `Training YOLO11n on ${dataset.images} explicitly reviewed localization images and ${dataset.boxes} cell boxes…`,
    log: [`Dataset: ${dataset.images} images / ${dataset.boxes} boxes`, `Split: ${dataset.trainImages} train images (${dataset.trainBoxes} boxes), ${dataset.validationImages} validation images (${dataset.validationBoxes} boxes)`, 'IMPORTANT: only explicitly saved full-image YOLO cell boxes and explicit true-background images are included. Crack-classifier feedback is excluded. Every visible PV cell in a positive YOLO source image must have a box; unboxed cells are treated as background.', `YOLO source images: ${getTrainingSourceRecords().map((record) => `${record.fileName} [${record.boxes.length} boxes; ${record.sampleType}]`).join(' | ')}`, `Settings: ${JSON.stringify(settings)}`, `Early stopping: ${settings.patience === 0 ? "disabled (full requested epoch count)" : `enabled with patience ${settings.patience}`}; manual Cancel remains available.`,
      ...(dataset.missedCellImages ? [`Missed-cell feedback: ${dataset.missedCells} operator-added cell(s) in ${dataset.missedCellImages} image(s); ${dataset.missedCopies} extra train copies added (validation not oversampled).`] : []),
      `Starting checkpoint: ${initialWeights ?? 'yolo11n.pt (base model)'}`,
      'Epoch analytics are collected after validation. Full worker output is saved to the downloadable log.'],
    epochMetrics: [], error: undefined, errorTraceback: undefined, startedAt: new Date().toISOString(), finishedAt: undefined,
    jobId, pid: undefined, ownerPid: process.pid, logFile, checkpointSource: initialWeights ?? 'yolo11n.pt' });
  const args = [script, '--data', dataset.yamlPath, '--epochs', String(epochs), '--patience', String(settings.patience), '--imgsz', String(imageSize), '--batch', String(batch),
    '--freeze', String(settings.freeze), '--optimizer', settings.optimizer, '--lr0', String(settings.lr0), '--lrf', String(settings.lrf),
    '--momentum', String(settings.momentum), '--weight-decay', String(settings.weightDecay), '--warmup-epochs', String(settings.warmupEpochs),
    '--warmup-momentum', String(settings.warmupMomentum), '--warmup-bias-lr', String(settings.warmupBiasLr),
    '--close-mosaic', String(settings.closeMosaic), '--mosaic', String(settings.mosaic), '--mixup', String(settings.mixup),
    '--degrees', String(settings.degrees), '--translate', String(settings.translate), '--scale', String(settings.scale),
    '--fliplr', String(settings.fliplr), '--flipud', String(settings.flipud), '--hsv-h', String(settings.hsvH),
    '--hsv-s', String(settings.hsvS), '--hsv-v', String(settings.hsvV), '--box-loss', String(settings.boxLoss),
    '--cls-loss', String(settings.clsLoss), '--dfl-loss', String(settings.dflLoss), '--multi-scale', String(settings.multiScale),
    '--seed', String(settings.seed), '--save-period', String(settings.savePeriod), '--min-map50', String(settings.minimumMap50), '--onnx-tolerance', String(settings.onnxGateTolerance),
    '--deterministic', String(settings.deterministic), '--cache', String(settings.cache), '--pretrained', String(settings.pretrained),
    '--cos-lr', String(settings.cosLr), '--project', RUNS_DIR, '--run-name', runName, '--activate', ACTIVE_MODEL,
    '--checkpoint-copy', STABLE_BEST_CHECKPOINT, '--state-file', JOB_STATE_FILE];
  if (initialWeights) args.push('--initial-weights', initialWeights);
  fs.rmSync(CONTROL_FILE, { force: true }); // never inherit a stale pause request
  try {
    const logFd = fs.openSync(logFile, 'a');
    try {
      activeChild = spawn(pythonExecutable(), args, { cwd: DATA_ROOT, windowsHide: true, stdio: ['ignore', logFd, logFd], env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8', YOLO_CONFIG_DIR: path.join(RUNTIME_DIR, 'ultralytics_config') } });
    } finally { fs.closeSync(logFd); }
    if (activeChild.pid) setState({ pid: activeChild.pid });
    // The background worker writes durable state; its stdout/stderr are also
    // redirected to the full per-job log file for post-mortem diagnostics.
    activeChild.unref();
    activeChild.on('error', (error) => {
      setState({ status: 'failed', message: 'YOLO training process could not start.', error: error.message, finishedAt: new Date().toISOString() });
      activeChild = null;
    });
    activeChild.on('close', (code, signal) => {
      activeChild = null;
      appendWorkerLog(`Worker exited: code=${code} signal=${signal ?? 'none'}`); // v1.4.12.6: visible in the saved worker log file
      pushLog(`Worker exited: code=${code} signal=${signal ?? 'none'}`);
      refreshStateFromDisk();
      if (code === 0 && fs.existsSync(ACTIVE_MODEL) && state.status === 'completed') {
        resetCellDetectorSession();
      } else if (state.status === 'training' || state.status === 'paused') {
        setState({ status: 'failed', message: 'YOLO training exited before completion was saved.', error: `Training worker exited with code ${code}${signal ? ` (signal ${signal})` : ''}. ${diagnoseWorkerDeath()} The active model was not replaced unless validation and export had completed.`, finishedAt: new Date().toISOString() });
      }
    });
  } catch (error) {
    activeChild = null;
    setState({ status: 'failed', message: 'Could not launch the YOLO training worker.', error: error instanceof Error ? error.message : String(error), finishedAt: new Date().toISOString() });
    throw error;
  }
  return getCellTrainingStatus();
}

export function cancelCellYoloTraining() {
  refreshStateFromDisk();
  if ((state.status === 'training' || state.status === 'paused') && state.pid) {
    fs.rmSync(CONTROL_FILE, { force: true });
    try {
      if (activeChild?.pid === state.pid) activeChild.kill();
      else process.kill(state.pid);
    } catch { /* worker may have exited between the status check and signal */ }
    setState({ status: 'cancelled', message: 'YOLO training cancelled by user.', error: 'Cancelled by user. The saved log, checkpoints and previous active model are retained.', finishedAt: new Date().toISOString() });
    activeChild = null;
  }
  return getCellTrainingStatus();
}

/** v1.4.10: pause = ask the worker (via a control file, works on Windows too) to wait at its next batch boundary. */
export function pauseCellYoloTraining() {
  refreshStateFromDisk();
  if (state.status === 'training' && state.pid) {
    fs.writeFileSync(CONTROL_FILE, 'pause', 'utf8');
    setState({ status: 'paused', message: 'Pausing YOLO training at the next batch… press Resume to continue.' });
    pushLog('Pause requested by the operator.');
  }
  return getCellTrainingStatus();
}
export function resumeCellYoloTraining() {
  refreshStateFromDisk();
  if (state.status === 'paused') {
    fs.rmSync(CONTROL_FILE, { force: true });
    setState({ status: 'training', message: 'YOLO training resumed.' });
    pushLog('Resume requested by the operator.');
  }
  return getCellTrainingStatus();
}


/**
 * Start the cell-localisation pipeline from zero while retaining the installed Python runtime and settings.
 * Only cell-specific annotations, generated YOLO artefacts, checkpoints and user ONNX models are removed.
 * The crack-classifier feedback/model is intentionally separate and remains untouched.
 */
export function resetCellTrainingDataAndModel() {
  refreshStateFromDisk();
  if (state.status === 'setting_up' || state.status === 'training' || state.status === 'paused' || (state.pid && pidIsAlive(state.pid))) {
    throw new Error('Cell training or runtime setup is still active. Finish or abort it before resetting the cell model.');
  }

  clearCellAnnotations();
  clearSavedCellReviewData();
  for (const directory of [DATASET_DIR, RUNS_DIR, LOGS_DIR]) fs.rmSync(directory, { recursive: true, force: true });
  for (const file of [ACTIVE_MODEL, PREVIOUS_MODEL, STABLE_BEST_CHECKPOINT, JOB_STATE_FILE, CONTROL_FILE]) fs.rmSync(file, { force: true });

  // A profile learned from the deleted cell boxes is no longer evidence-backed. Restore only that profile
  // to the normal default while preserving every other operator training/inference setting.
  const settings = getTrainingSettings();
  const profiles = settings.cellTypeFilter.profiles.filter((profile) => profile.id !== 'learned');
  const hasStandard = profiles.some((profile) => profile.id === 'standard');
  const safeProfiles = hasStandard ? profiles.map((profile) => profile.id === 'standard' ? { ...profile, enabled: true } : profile) : [
    { id: 'standard', name: 'Standard cell (default)', mode: 'accept' as const, enabled: true, aspect: 1.0, aspectTolerance: 0.3, minLongPx: 0, maxLongPx: 0 },
    ...profiles,
  ];
  saveTrainingSettings({ ...settings, cellTypeFilter: { ...settings.cellTypeFilter, profiles: safeProfiles } });

  resetCellDetectorSession();
  activeChild = null;
  state = defaultState();
  persistState();
  return getCellTrainingStatus();
}

export function rollbackCellModel() {
  if (!fs.existsSync(PREVIOUS_MODEL)) throw new Error('No previous cell model is available to roll back to.');
  fs.mkdirSync(MODEL_DIR, { recursive: true });
  const temporary = ACTIVE_MODEL + '.rollback.tmp';
  fs.copyFileSync(PREVIOUS_MODEL, temporary);
  if (fs.existsSync(ACTIVE_MODEL)) fs.copyFileSync(ACTIVE_MODEL, PREVIOUS_MODEL);
  fs.renameSync(temporary, ACTIVE_MODEL);
  resetCellDetectorSession();
  return getCellTrainingStatus();
}
