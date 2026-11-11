/** Durable progress, logs and epoch analytics for cell-level crack-classifier training. */
import fs from 'node:fs';
import path from 'node:path';
import { DATA_ROOT } from './dataPaths.ts';

import type { TrainControl } from './cellCnn.ts';

export type CrackJobStatus = 'idle' | 'preparing' | 'training' | 'validating' | 'paused' | 'completed' | 'failed';
export interface CrackEpochMetric {
  epoch: number;
  trainLoss: number;
  validationLoss: number;
  precision: number;
  recall: number;
  specificity: number;
  balancedAccuracy: number;
  f1: number;
  component: 'CNN';
}
export interface CrackTrainingJobState {
  status: CrackJobStatus;
  jobId?: string;
  pid?: number;
  progress: number;
  stage: string;
  message: string;
  algorithm?: string;
  epoch?: number;
  epochs?: number;
  tree?: number;
  trees?: number;
  samples?: number;
  trainingSamples?: number;
  validationSamples?: number;
  metrics: CrackEpochMetric[];
  log: string[];
  startedAt?: string;
  finishedAt?: string;
  updatedAt?: string;
  error?: string;
  errorTraceback?: string;
  report?: Record<string, any>;
}

const JOB_DIR = path.join(DATA_ROOT, 'training', 'crack_classifier');
const JOB_FILE = path.join(JOB_DIR, 'job-state.json');
const LOG_FILE = path.join(JOB_DIR, 'training.log');
const CSV_FILE = path.join(JOB_DIR, 'epoch-metrics.csv');
const activeStatuses: CrackJobStatus[] = ['preparing', 'training', 'validating', 'paused'];
const emptyState = (): CrackTrainingJobState => ({ status: 'idle', progress: 0, stage: 'idle', message: 'No crack-classifier training run has started.', metrics: [], log: [] });
function readState(): CrackTrainingJobState {
  try {
    const parsed = JSON.parse(fs.readFileSync(JOB_FILE, 'utf8')) as CrackTrainingJobState;
    if (!parsed || !['idle', 'preparing', 'training', 'validating', 'paused', 'completed', 'failed'].includes(parsed.status)) return emptyState();
    return { ...parsed, metrics: Array.isArray(parsed.metrics) ? parsed.metrics.slice(-1000).map((row: any) => ({ ...row, f1: Number.isFinite(row.f1) ? row.f1 : Number(((2 * (row.precision || 0) * (row.recall || 0)) / Math.max(1e-12, (row.precision || 0) + (row.recall || 0))).toFixed(4)) })) : [], log: Array.isArray(parsed.log) ? parsed.log.slice(-1000) : [] };
  } catch { return emptyState(); }
}
let state = readState();
function pidAlive(pid?: number): boolean {
  if (!pid || !Number.isInteger(pid) || pid < 1) return false;
  try { process.kill(pid, 0); return true; } catch (error) { return (error as NodeJS.ErrnoException).code === 'EPERM'; }
}
function persist(): void {
  fs.mkdirSync(JOB_DIR, { recursive: true });
  state.updatedAt = new Date().toISOString();
  const temp = `${JOB_FILE}.tmp`;
  fs.writeFileSync(temp, JSON.stringify(state, null, 2));
  try { fs.renameSync(temp, JOB_FILE); }
  catch { fs.rmSync(JOB_FILE, { force: true }); fs.renameSync(temp, JOB_FILE); }
  fs.writeFileSync(LOG_FILE, `${state.log.join('\n')}${state.log.length ? '\n' : ''}`);
  const columns = ['epoch', 'train_loss', 'validation_loss', 'precision', 'recall', 'specificity', 'balanced_accuracy', 'f1', 'component'];
  const q = (v: unknown) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  fs.writeFileSync(CSV_FILE, `${columns.join(',')}\r\n${state.metrics.map((row) => columns.map((key) => q((row as any)[key === 'train_loss' ? 'trainLoss' : key === 'validation_loss' ? 'validationLoss' : key === 'balanced_accuracy' ? 'balancedAccuracy' : key])).join(',')).join('\r\n')}${state.metrics.length ? '\r\n' : ''}`);
}
function recoverInterruptedJob(): void {
  if (activeStatuses.includes(state.status) && !pidAlive(state.pid)) {
    state = { ...state, status: 'failed', stage: 'interrupted', message: 'Training stopped when the server process exited.', error: 'The saved job was recovered after an application restart. The previously active crack model was not replaced; the last completed epoch log and CSV are retained.', finishedAt: new Date().toISOString() };
    persist();
  }
}
recoverInterruptedJob();

// ---- v1.4.10 pause / resume / abort (cooperative: honoured at batch, tree and crop-extraction boundaries) ----
export const CRACK_ABORT_MESSAGE = 'Crack-classifier training was aborted by the operator.';
let pauseRequested = false, abortRequested = false, resumeStatus: CrackJobStatus = 'training';
let waiters: Array<() => void> = [];
function releaseWaiters(): void { const w = waiters; waiters = []; w.forEach((resolve) => resolve()); }
export const crackTrainingControl: TrainControl = {
  waitIfPaused: async () => {
    if (abortRequested) throw new Error(CRACK_ABORT_MESSAGE);
    while (pauseRequested) { await new Promise<void>((resolve) => waiters.push(resolve)); if (abortRequested) throw new Error(CRACK_ABORT_MESSAGE); }
  },
};
export function pauseCrackTraining(): CrackTrainingJobState {
  if (activeStatuses.includes(state.status) && state.status !== 'paused') {
    pauseRequested = true; resumeStatus = state.status;
    state = { ...state, status: 'paused', message: 'Training paused. Press Resume to continue from this point.' };
    appendCrackTrainingLog('Training paused by the operator.');
  }
  return getCrackTrainingJobState();
}
export function resumeCrackTraining(): CrackTrainingJobState {
  if (state.status === 'paused') {
    pauseRequested = false;
    state = { ...state, status: resumeStatus, message: 'Training resumed.' };
    appendCrackTrainingLog('Training resumed by the operator.');
    releaseWaiters();
  }
  return getCrackTrainingJobState();
}
export function abortCrackTraining(): CrackTrainingJobState {
  if (activeStatuses.includes(state.status)) { abortRequested = true; pauseRequested = false; releaseWaiters(); appendCrackTrainingLog('Abort requested by the operator.'); }
  return getCrackTrainingJobState();
}

export function getCrackTrainingJobState(): CrackTrainingJobState { return { ...state, metrics: [...state.metrics], log: [...state.log] }; }
export function isCrackTrainingActive(): boolean { return activeStatuses.includes(state.status); }
export function beginCrackTrainingJob(jobId: string, algorithm: string, epochs: number, message: string): CrackTrainingJobState {
  pauseRequested = false; abortRequested = false; releaseWaiters();
  state = { status: 'preparing', jobId, pid: process.pid, progress: 1, stage: 'preparing', message, algorithm, epochs, epoch: 0, tree: 0, trees: 0, metrics: [], log: [], startedAt: new Date().toISOString() };
  appendCrackTrainingLog(`Started ${algorithm} crack-classifier training; configured CNN epochs: ${epochs}.`);
  persist();
  return getCrackTrainingJobState();
}
export function updateCrackTrainingJob(patch: Partial<CrackTrainingJobState>): CrackTrainingJobState {
  if (patch.status === 'completed' || patch.status === 'failed') { pauseRequested = false; releaseWaiters(); }
  state = { ...state, ...patch };
  if (pauseRequested) { if (patch.status && patch.status !== 'paused') resumeStatus = patch.status; state.status = 'paused'; } // progress events must not un-pause
  persist();
  return getCrackTrainingJobState();
}
export function appendCrackTrainingLog(line: string): void {
  const stamped = `[${new Date().toLocaleTimeString('en-GB', { hour12: false })}] ${line}`;
  state = { ...state, log: [...state.log.slice(-999), stamped] };
  persist();
}
export function addCrackEpochMetric(row: CrackEpochMetric): void {
  const rest = state.metrics.filter((item) => item.epoch !== row.epoch);
  state = { ...state, metrics: [...rest, row].sort((a, b) => a.epoch - b.epoch) };
  persist();
}
export function crackTrainingMetricsCsv(): string {
  const columns = ['epoch', 'train_loss', 'validation_loss', 'precision', 'recall', 'specificity', 'balanced_accuracy', 'f1', 'component'];
  const q = (v: unknown) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  return `${columns.join(',')}\r\n${state.metrics.map((row) => [row.epoch, row.trainLoss, row.validationLoss, row.precision, row.recall, row.specificity, row.balancedAccuracy, row.f1, row.component].map(q).join(',')).join('\r\n')}${state.metrics.length ? '\r\n' : ''}`;
}
export function crackTrainingLogText(): string { return `${state.log.join('\n')}${state.log.length ? '\n' : ''}`; }
