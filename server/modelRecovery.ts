/**
 * v1.4.17.7: recover trained models from earlier releases.
 * Why: the one-time legacy migration only looks at sibling release folders inside the Working Folder, copies a file only when it is newer,
 * remembers what it has "seen" forever, and copies the .onnx and .json of a pair independently. A model trained in an older release that
 * lives anywhere else (another data folder, the per-user data folder, an old release's own data/ folder), or whose copy was removed later,
 * was therefore never picked up and the app reported "no model". This pass runs on every start and on demand: it finds COMPLETE, valid
 * model pairs in every known location and installs the newest one into the active models folder when that folder has no valid pair or an older one.
 * Source files are never modified or deleted. A pair is copied .onnx first, .json last (the app treats a model without its meta as absent).
 */
import fs from 'node:fs';
import path from 'node:path';
import { APP_ROOT, DATA_ROOT, MODEL_DATA_DIR, WORKING_FOLDER, discoverLegacyReleaseRoots } from './dataPaths.ts';
import { userConfigDir } from './appConfig.ts';
import { resetCellClassifierSession } from './cellClassifier.ts';

export type RecoverableModel = 'anomaly' | 'classifier' | 'cell';
const PAIRS: Record<'anomaly' | 'classifier', { onnx: string; meta: string; schema: string }> = {
  anomaly: { onnx: 'cellAnomaly.onnx', meta: 'cellAnomaly.json', schema: 'cell-anomaly-v1' },
  classifier: { onnx: 'cellClassifier.onnx', meta: 'cellClassifier.json', schema: 'cell-classifier-v1' },
};
export interface FoundModel { model: RecoverableModel; dir: string; trainedAt: number; version?: string; onnxBytes: number }
export interface RecoveryReport {
  ranAt: string; searched: string[]; found: FoundModel[];
  installed: Array<{ model: RecoverableModel; from: string; version?: string }>;
  kept: Array<{ model: RecoverableModel; reason: string }>; warnings: string[];
}
let lastReport: RecoveryReport | null = null;
export const getLastRecoveryReport = (): RecoveryReport | null => lastReport;

const exists = (p: string): boolean => { try { return fs.statSync(p).isFile(); } catch { return false; } };
const looksOnnx = (file: string): boolean => {
  try { const st = fs.statSync(file); if (!st.isFile() || st.size < 1024) return false; const fd = fs.openSync(file, 'r'); const b = Buffer.alloc(1); try { fs.readSync(fd, b, 0, 1, 0); } finally { fs.closeSync(fd); } return b[0] === 0x08; } catch { return false; }
};
/** A pair is usable only when both files exist, the ONNX is an ONNX file, and the meta passes the same range checks the loader applies. */
export function readValidPair(dir: string, model: 'anomaly' | 'classifier'): { trainedAt: number; version?: string; onnxBytes: number } | null {
  const spec = PAIRS[model], onnx = path.join(dir, spec.onnx), meta = path.join(dir, spec.meta);
  if (!exists(onnx) || !exists(meta) || !looksOnnx(onnx)) return null;
  try {
    const raw = JSON.parse(fs.readFileSync(meta, 'utf8'));
    const ok = raw?.schema === spec.schema && Number.isInteger(raw.imgsz) && raw.imgsz >= 32 && raw.imgsz <= 1024 && Number.isInteger(raw.crackedIndex) && raw.crackedIndex >= 0 &&
      Number.isFinite(raw.threshold) && raw.threshold > 0 && raw.threshold < 1 && Number.isFinite(raw.cellBuffer) && raw.cellBuffer >= 0 && raw.cellBuffer <= 0.5;
    if (!ok) return null;
    const stamped = Date.parse(raw.trainedAt ?? '');
    return { trainedAt: Number.isFinite(stamped) ? stamped : fs.statSync(meta).mtimeMs, version: typeof raw.version === 'string' ? raw.version : undefined, onnxBytes: fs.statSync(onnx).size };
  } catch { return null; }
}

/** Every folder that could hold models written by an earlier release (deduplicated, active folder first). */
export function candidateModelDirs(): string[] {
  const out = new Set<string>([path.resolve(MODEL_DATA_DIR)]);
  const add = (p: string | undefined) => { if (p) out.add(path.resolve(p)); };
  const dataRoots = new Set<string>([DATA_ROOT, path.join(WORKING_FOLDER, 'data'), path.join(path.dirname(APP_ROOT), 'data'), path.join(path.dirname(path.dirname(APP_ROOT)), 'data')]);
  try { dataRoots.add(path.join(userConfigDir(), 'data')); } catch { /* no user profile */ }
  let releases: string[] = [];
  try { releases = discoverLegacyReleaseRoots(); } catch { /* unreadable working folder */ }
  // Releases extracted next to this one but outside the Working Folder scan (e.g. in the parent of this folder).
  try {
    const parent = path.dirname(APP_ROOT);
    for (const e of fs.readdirSync(parent, { withFileTypes: true })) if (e.isDirectory() && /^PV[_-]?CrackVision/i.test(e.name)) releases.push(path.join(parent, e.name));
  } catch { /* parent not listable */ }
  for (const r of releases) { add(path.join(r, 'server', 'models')); add(path.join(r, 'data', 'models')); }
  for (const d of dataRoots) add(path.join(d, 'models'));
  return [...out];
}

function installPair(srcDir: string, model: 'anomaly' | 'classifier'): void {
  const spec = PAIRS[model];
  fs.mkdirSync(MODEL_DATA_DIR, { recursive: true });
  for (const name of [spec.onnx, spec.meta]) { // onnx first, meta last
    const dest = path.join(MODEL_DATA_DIR, name), tmp = `${dest}.recover.tmp`;
    if (exists(dest)) { try { fs.copyFileSync(dest, `${dest}.before-recovery`); } catch { /* backup is best effort */ } }
    fs.copyFileSync(path.join(srcDir, name), tmp);
    fs.renameSync(tmp, dest);
  }
}

export function recoverPreviousModels(): RecoveryReport {
  const report: RecoveryReport = { ranAt: new Date().toISOString(), searched: [], found: [], installed: [], kept: [], warnings: [] };
  try {
    const dirs = candidateModelDirs(); report.searched = dirs.filter((d) => { try { return fs.statSync(d).isDirectory(); } catch { return false; } });
    const active = path.resolve(MODEL_DATA_DIR);
    for (const model of ['anomaly', 'classifier'] as const) {
      const found = report.searched.map((dir) => { const p = readValidPair(dir, model); return p ? { model, dir, ...p } as FoundModel : null; }).filter((x): x is FoundModel => x !== null);
      report.found.push(...found);
      const current = found.find((f) => path.resolve(f.dir) === active);
      const best = found.filter((f) => path.resolve(f.dir) !== active).sort((a, b) => b.trainedAt - a.trainedAt)[0];
      if (!best) { report.kept.push({ model, reason: current ? 'the active model is already the newest found' : 'no earlier trained model found' }); continue; }
      if (current && current.trainedAt >= best.trainedAt) { report.kept.push({ model, reason: 'the active model is newer than every earlier one' }); continue; }
      try { installPair(best.dir, model); report.installed.push({ model, from: best.dir, version: best.version }); }
      catch (error) { report.warnings.push(`Could not install the ${model} model from ${best.dir}: ${error instanceof Error ? error.message : String(error)}`); }
    }
    // Cell detector: only fill a gap (an existing cell.onnx has its own roll-back flow and is never replaced here).
    const activeCell = path.join(MODEL_DATA_DIR, 'cell.onnx');
    if (!looksOnnx(activeCell)) {
      const cells = report.searched.filter((d) => path.resolve(d) !== active && looksOnnx(path.join(d, 'cell.onnx'))).map((d) => ({ d, t: fs.statSync(path.join(d, 'cell.onnx')).mtimeMs })).sort((a, b) => b.t - a.t);
      for (const c of cells) report.found.push({ model: 'cell', dir: c.d, trainedAt: c.t, onnxBytes: fs.statSync(path.join(c.d, 'cell.onnx')).size });
      if (cells[0]) {
        try { fs.mkdirSync(MODEL_DATA_DIR, { recursive: true }); const tmp = `${activeCell}.recover.tmp`; fs.copyFileSync(path.join(cells[0].d, 'cell.onnx'), tmp); fs.renameSync(tmp, activeCell); report.installed.push({ model: 'cell', from: cells[0].d }); }
        catch (error) { report.warnings.push(`Could not install the cell detector from ${cells[0].d}: ${error instanceof Error ? error.message : String(error)}`); }
      }
    }
    if (report.installed.length) resetCellClassifierSession();
  } catch (error) { report.warnings.push(error instanceof Error ? error.message : String(error)); }
  lastReport = report;
  if (report.installed.length) console.log(`[Model recovery] installed ${report.installed.map((i) => `${i.model} <- ${i.from}`).join('; ')}`);
  return report;
}
