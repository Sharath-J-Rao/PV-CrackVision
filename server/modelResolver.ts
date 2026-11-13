/**
 * v1.4.13.1: offline model resolution. Order: the user's own model (trained/imported, in the data folder) -> the model
 * bundled with the app (resources/models, read-only, integrity + version checked) -> none (classical fallback).
 * A missing/corrupt/incompatible candidate is skipped with a warning; files are never modified or deleted.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { APP_ROOT, MODEL_DATA_DIR } from './dataPaths.ts';
import { APP_VERSION } from './appVersion.ts';
import { isPackagedApp } from './appConfig.ts';

export const CELL_MODEL_FILE = 'cell.onnx';
export const CRACK_MODEL_FILE = 'cellCrackModel.user.json';
export type ModelSource = 'user' | 'bundled';
export type ModelCandidate = { path: string; source: ModelSource };
export type ModelResolution = { candidates: ModelCandidate[]; warnings: string[] };
type ManifestEntry = { sha256: string; size: number; schema?: string };
type Manifest = { modelVersion?: string; minAppVersion?: string; maxAppVersion?: string; files?: Record<string, ManifestEntry> };

export const bundledModelsDir = (env: NodeJS.ProcessEnv = process.env): string =>
  path.resolve(env.PV_BUNDLED_MODELS_DIR?.trim() || (isPackagedApp(env) && (process as any).resourcesPath ? path.join((process as any).resourcesPath, 'models') : path.join(APP_ROOT, 'resources', 'models')));

const versionParts = (v: string): number[] => (v.match(/\d+/g) ?? []).map(Number);
/** Numeric compare of release labels ("1.4.13-1" == "1.4.13.1"). Missing parts count as 0. */
export function compareVersions(a: string, b: string): number {
  const x = versionParts(a), y = versionParts(b);
  for (let i = 0; i < Math.max(x.length, y.length); i++) { const d = (x[i] ?? 0) - (y[i] ?? 0); if (d) return Math.sign(d); }
  return 0;
}

const hashCache = new Map<string, string>();
function sha256(file: string, stat: fs.Stats): string {
  const key = `${file}:${stat.size}:${stat.mtimeMs}`;
  let digest = hashCache.get(key);
  if (!digest) { digest = crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'); hashCache.set(key, digest); }
  return digest;
}

function readManifest(dir: string): Manifest | null {
  try { const m = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json'), 'utf8')); return m && typeof m === 'object' ? m as Manifest : null; } catch { return null; }
}

/** Cheap structural check (no inference runtime needed). The ONNX test-load happens when the session is created. */
function structurallyValid(name: string, file: string, stat: fs.Stats): string | null {
  if (!stat.isFile() || stat.size < (name.endsWith('.onnx') ? 1024 : 2)) return 'file is missing its content or too small';
  if (name.endsWith('.onnx')) { // cell detector and cell-crop classifier are both ONNX
    const head = Buffer.alloc(1); const fd = fs.openSync(file, 'r');
    try { fs.readSync(fd, head, 0, 1, 0); } finally { fs.closeSync(fd); }
    return head[0] === 0x08 ? null : 'not an ONNX model file';
  }
  try { const raw = JSON.parse(fs.readFileSync(file, 'utf8')); return raw && typeof raw.schema === 'string' ? null : 'not a crack-classifier JSON'; } catch { return 'unreadable JSON'; }
}

export function checkBundledFile(name: string, dir: string, appVersion = APP_VERSION): { ok: boolean; reason?: string; path: string } {
  const file = path.join(dir, name);
  if (!fs.existsSync(file)) return { ok: false, reason: 'not bundled', path: file };
  const manifest = readManifest(dir);
  const entry = manifest?.files?.[name];
  if (!manifest || !entry) return { ok: false, reason: 'manifest.json is missing or has no entry for this file', path: file };
  if (manifest.minAppVersion && compareVersions(appVersion, manifest.minAppVersion) < 0) return { ok: false, reason: `needs app ${manifest.minAppVersion} or newer (this is ${appVersion})`, path: file };
  if (manifest.maxAppVersion && compareVersions(appVersion, manifest.maxAppVersion) > 0) return { ok: false, reason: `built for app ${manifest.maxAppVersion} or older (this is ${appVersion})`, path: file };
  let stat: fs.Stats;
  try { stat = fs.statSync(file); } catch { return { ok: false, reason: 'cannot be read', path: file }; }
  if (stat.size !== entry.size) return { ok: false, reason: `size ${stat.size} differs from manifest ${entry.size} (corrupt or incomplete)`, path: file };
  const bad = structurallyValid(name, file, stat);
  if (bad) return { ok: false, reason: bad, path: file };
  try { if (sha256(file, stat) !== entry.sha256) return { ok: false, reason: 'checksum differs from manifest (corrupt or modified)', path: file }; } catch { return { ok: false, reason: 'cannot be read', path: file }; }
  return { ok: true, path: file };
}

export function resolveModel(name: string, userDir: string = MODEL_DATA_DIR, bundledDir: string = bundledModelsDir(), appVersion = APP_VERSION): ModelResolution {
  const candidates: ModelCandidate[] = [], warnings: string[] = [];
  const userFile = path.join(userDir, name);
  if (fs.existsSync(userFile)) {
    try {
      const bad = structurallyValid(name, userFile, fs.statSync(userFile));
      if (bad) warnings.push(`Your ${name} was skipped: ${bad}.`); else candidates.push({ path: userFile, source: 'user' });
    } catch { warnings.push(`Your ${name} could not be read and was skipped.`); }
  }
  const bundled = checkBundledFile(name, bundledDir, appVersion);
  if (bundled.ok) candidates.push({ path: bundled.path, source: 'bundled' });
  else if (bundled.reason !== 'not bundled') warnings.push(`Bundled ${name} was skipped: ${bundled.reason}.`);
  return { candidates, warnings };
}

export const CLASSIFIER_ONNX_FILE = 'cellClassifier.onnx';
export const CLASSIFIER_META_FILE = 'cellClassifier.json';
/** The cell-crop classifier is a pair (.onnx + .json): a complete user pair wins, otherwise the verified bundled pair, otherwise null. */
export function resolveClassifierFiles(userDir: string = MODEL_DATA_DIR, bundledDir: string = bundledModelsDir(), appVersion = APP_VERSION): { onnx: string; meta: string; source: ModelSource } | null {
  const user = [CLASSIFIER_ONNX_FILE, CLASSIFIER_META_FILE].map((n) => path.join(userDir, n));
  if (user.every((f) => fs.existsSync(f))) return { onnx: user[0], meta: user[1], source: 'user' };
  const bundled = [CLASSIFIER_ONNX_FILE, CLASSIFIER_META_FILE].map((n) => checkBundledFile(n, bundledDir, appVersion));
  return bundled.every((b) => b.ok) ? { onnx: bundled[0].path, meta: bundled[1].path, source: 'bundled' } : null;
}

export const resolveCellModel =(): ModelResolution => resolveModel(CELL_MODEL_FILE);
export const resolveCrackModelFile = (): ModelResolution => resolveModel(CRACK_MODEL_FILE);

/** Summary for /api/health, the detector status and the smoke test. */
export function getModelHealth() {
  const summarize = (r: ModelResolution) => ({ ok: r.candidates.length > 0, source: r.candidates[0]?.source ?? null, warnings: r.warnings });
  return { cell: summarize(resolveCellModel()), crack: summarize(resolveCrackModelFile()), bundledDir: bundledModelsDir(), manifest: readManifest(bundledModelsDir())?.modelVersion ?? null };
}
