/**
 * Shared writable data locations for every PV CrackVision release extracted under
 * the same Working Folder. Source code stays in each release directory; mutable
 * annotations, datasets, logs, checkpoints, models and runtimes live in ../data.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { configuredDataRoot, isPackagedApp, userConfigDir } from './appConfig.ts';

export const APP_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
function findWorkingFolder(start: string): string {
  let current = path.resolve(start);
  while (true) {
    if (path.basename(current).toLowerCase() === 'working folder') return current;
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return path.resolve(start, '..');
}
export const WORKING_FOLDER = findWorkingFolder(APP_ROOT);
/**
 * Resolution order (v1.4.8.0): PV_CRACKVISION_DATA_DIR env > folder chosen in the app (per-user config, survives
 * upgrades) > default. The default is Working Folder/data for source/ZIP installs (unchanged, so existing data is found)
 * and <user config dir>/data for a packaged .exe.
 */
const ENV_DATA_ROOT = process.env.PV_CRACKVISION_DATA_DIR?.trim();
const CONFIGURED_DATA_ROOT = configuredDataRoot();
export const DATA_ROOT_SOURCE: 'env' | 'config' | 'default' = ENV_DATA_ROOT ? 'env' : CONFIGURED_DATA_ROOT ? 'config' : 'default';
export const DEFAULT_DATA_ROOT = isPackagedApp() ? path.join(userConfigDir(), 'data') : path.join(WORKING_FOLDER, 'data');
export const DATA_ROOT = path.resolve(ENV_DATA_ROOT || CONFIGURED_DATA_ROOT || DEFAULT_DATA_ROOT);
export const DATA_TRAINING_DIR = path.join(DATA_ROOT, 'training');
export const FEEDBACK_DIR = path.join(DATA_TRAINING_DIR, 'feedback');
export const CELL_ANNOTATIONS_DIR = path.join(DATA_TRAINING_DIR, 'cell_annotations');
export const CELL_YOLO_DIR = path.join(DATA_TRAINING_DIR, 'cell_yolo');
export const TRAINING_MODELS_DIR = path.join(DATA_TRAINING_DIR, 'models');
export const RUNTIME_DIR = path.join(DATA_ROOT, 'runtime');
export const VENV_DIR = path.join(RUNTIME_DIR, 'cell-yolo-venv');
export const EMBEDDED_PYTHON_DIR = path.join(RUNTIME_DIR, 'python-runtime');
export const MODEL_DATA_DIR = path.join(DATA_ROOT, 'models');
export const JOB_STATE_FILE = path.join(DATA_TRAINING_DIR, '.cell-yolo-job-state.json');
export const RUNTIME_MARKER = path.join(RUNTIME_DIR, '.cell-yolo-ready');

const MIGRATION_FILE = path.join(DATA_ROOT, '.migration-state-v1.json');
type MigrationState = { seen: Record<string, { size: number; mtimeMs: number }> };
const migrationState: MigrationState = (() => {
  try {
    const parsed = JSON.parse(fs.readFileSync(MIGRATION_FILE, 'utf8')) as MigrationState;
    return parsed && typeof parsed.seen === 'object' ? parsed : { seen: {} };
  } catch { return { seen: {} }; }
})();

function boxKey(box: any): string {
  return ['left', 'top', 'width', 'height'].map((key) => Number(box?.[key]).toFixed(3)).join(':');
}

export function mergeMigrationJsonArray(sourceText: string, destinationText: string): string | null {
  try {
    const source = JSON.parse(sourceText);
    const destination = JSON.parse(destinationText);
    if (!Array.isArray(source) || !Array.isArray(destination)) return null;
    const merged = destination.map((item) => ({ ...item }));
    for (const incoming of source) {
      const identity = incoming?.boxes && incoming?.imageFile
        ? `image:${incoming.imageFile}`
        : incoming?.id ? `id:${incoming.id}` : null;
      if (!identity) { merged.push(incoming); continue; }
      const index = merged.findIndex((item) => (item?.boxes && item?.imageFile
        ? `image:${item.imageFile}` : item?.id ? `id:${item.id}` : null) === identity);
      if (index < 0) { merged.push(incoming); continue; }
      const current = merged[index];
      const newer = Date.parse(incoming?.createdAt ?? '') > Date.parse(current?.createdAt ?? '');
      if (Array.isArray(incoming?.boxes) && Array.isArray(current?.boxes)) {
        const byBox = new Map<string, any>();
        for (const box of current.boxes) byBox.set(boxKey(box), box);
        for (const box of incoming.boxes) byBox.set(boxKey(box), box);
        merged[index] = { ...(newer ? current : incoming), ...(newer ? incoming : current), boxes: [...byBox.values()] };
      } else if (newer) merged[index] = { ...current, ...incoming };
    }
    return JSON.stringify(merged, null, 2);
  } catch { return null; }
}

export function rebaseMigratedStatePaths(text: string, legacyRoot: string): string {
  const replacements: Array<[string, string]> = [
    [path.join(legacyRoot, 'training', 'cell_yolo'), CELL_YOLO_DIR],
    [path.join(legacyRoot, 'training', 'models'), TRAINING_MODELS_DIR],
    [path.join(legacyRoot, 'training', '.cell-yolo-venv'), VENV_DIR],
    [path.join(legacyRoot, 'training', '.python-runtime'), EMBEDDED_PYTHON_DIR],
    [path.join(legacyRoot, 'server', 'models'), MODEL_DATA_DIR],
    [path.join(legacyRoot, 'training'), DATA_TRAINING_DIR],
    [legacyRoot, APP_ROOT],
  ];
  let result = text;
  for (const [from, to] of replacements) {
    const escaped = from.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/[\\/]+/g, '[\\\\/]');
    result = result.replace(new RegExp(escaped, 'gi'), to);
  }
  return result;
}

let importedFileCount = 0;

function copyChangedFile(source: string, destination: string, legacyRoot: string, manifestKey: string): void {
  const stat = fs.statSync(source);
  const signature = { size: stat.size, mtimeMs: stat.mtimeMs };
  const prior = migrationState.seen[manifestKey];
  if (prior && prior.size === signature.size && prior.mtimeMs === signature.mtimeMs) return;
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  let sourceText: string | undefined;
  if (source.endsWith('.json')) {
    try { sourceText = fs.readFileSync(source, 'utf8'); } catch { /* binary or unreadable JSON is copied as bytes below */ }
  }
  if (sourceText !== undefined && path.basename(source) === '.cell-yolo-job-state.json') {
    if (!fs.existsSync(destination) || stat.mtimeMs > fs.statSync(destination).mtimeMs) {
      fs.writeFileSync(destination, rebaseMigratedStatePaths(sourceText, legacyRoot), 'utf8');
      importedFileCount++;
    }
  } else if (sourceText !== undefined && fs.existsSync(destination)) {
    let destinationText: string | undefined;
    try { destinationText = fs.readFileSync(destination, 'utf8'); } catch { /* replace non-text destination */ }
    const merged = destinationText === undefined ? null : mergeMigrationJsonArray(sourceText, destinationText);
    if (merged !== null) {
      fs.writeFileSync(destination, merged, 'utf8');
      fs.utimesSync(destination, stat.atime, stat.mtime);
      importedFileCount++;
    } else if (stat.mtimeMs > fs.statSync(destination).mtimeMs) {
      fs.copyFileSync(source, destination);
      fs.utimesSync(destination, stat.atime, stat.mtime);
      importedFileCount++;
    }
  } else if (!fs.existsSync(destination) || stat.mtimeMs > fs.statSync(destination).mtimeMs) {
    fs.copyFileSync(source, destination);
    fs.utimesSync(destination, stat.atime, stat.mtime);
    importedFileCount++;
  }
  migrationState.seen[manifestKey] = signature;
}

function copyTree(source: string, destination: string, legacyRoot: string, manifestPrefix: string): void {
  if (!fs.existsSync(source)) return;
  const entries = fs.readdirSync(source, { withFileTypes: true });
  for (const entry of entries) {
    const src = path.join(source, entry.name);
    const dst = path.join(destination, entry.name);
    if (entry.isSymbolicLink()) continue;
    if (entry.isDirectory()) copyTree(src, dst, legacyRoot, `${manifestPrefix}/${entry.name}`);
    else if (entry.isFile()) copyChangedFile(src, dst, legacyRoot, `${manifestPrefix}/${entry.name}`);
  }
}

export function discoverLegacyReleaseRoots(workingFolder = WORKING_FOLDER, currentRoot = APP_ROOT): string[] {
  const candidates = new Set<string>([currentRoot]);
  const releaseName = /^PV[_-]?CrackVision/i;
  const scan = (directory: string, depth: number) => {
    let entries: fs.Dirent[];
    try { entries = fs.readdirSync(directory, { withFileTypes: true }); }
    catch { return; }
    for (const entry of entries) {
      if (!entry.isDirectory() || !releaseName.test(entry.name)) continue;
      const candidate = path.join(directory, entry.name);
      if (fs.existsSync(path.join(candidate, 'server.ts')) && fs.existsSync(path.join(candidate, 'server')) && fs.existsSync(path.join(candidate, 'training'))) {
        candidates.add(candidate);
      } else if (depth < 4) {
        // Some release ZIPs have a named wrapper folder containing a second
        // PV_CrackVision_* project folder. Walk only matching wrapper names.
        scan(candidate, depth + 1);
      }
    }
  };
  scan(workingFolder, 0);
  return [...candidates].filter((candidate) => path.resolve(candidate) !== path.resolve(DATA_ROOT));
}

/** Existing per-release runtimes are fallback candidates if a copied Windows venv is not relocatable. */
export function legacyTrainingPythonCandidates(platform: NodeJS.Platform = process.platform): string[] {
  const relativePaths = platform === 'win32'
    ? [path.join('training', '.python-runtime', 'python.exe'), path.join('training', '.cell-yolo-venv', 'Scripts', 'python.exe')]
    : [path.join('training', '.cell-yolo-venv', 'bin', 'python')];
  return discoverLegacyReleaseRoots().filter((releaseRoot) => path.resolve(releaseRoot) !== path.resolve(APP_ROOT))
    .flatMap((releaseRoot) => relativePaths.map((relative) => path.join(releaseRoot, relative)))
    .filter((candidate) => fs.existsSync(candidate));
}

/**
 * Import user artifacts from all sibling releases. Copies are merged and source
 * files are never deleted. A manifest avoids repeatedly copying an unchanged
 * multi-GB Python environment on every application start.
 */
export function migrateLegacyTrainingData(): { dataRoot: string; scannedReleases: number; importedFiles: number } {
  fs.mkdirSync(DATA_ROOT, { recursive: true });
  importedFileCount = 0;
  const mappings: Array<[string, string, string]> = [
    ['training/feedback', path.join(DATA_TRAINING_DIR, 'feedback'), 'training/feedback'],
    ['training/cell_annotations', path.join(DATA_TRAINING_DIR, 'cell_annotations'), 'training/cell_annotations'],
    ['training/cell_yolo', CELL_YOLO_DIR, 'training/cell_yolo'],
    ['training/models', TRAINING_MODELS_DIR, 'training/models'],
    ['training/.cell-yolo-venv', VENV_DIR, 'training/.cell-yolo-venv'],
    ['training/.python-runtime', EMBEDDED_PYTHON_DIR, 'training/.python-runtime'],
    ['training/ultralytics_config', path.join(RUNTIME_DIR, 'ultralytics_config'), 'training/ultralytics_config'],
    ['server/models', MODEL_DATA_DIR, 'server/models'],
  ];
  const releaseRoots = discoverLegacyReleaseRoots();
  for (const legacyRoot of releaseRoots) {
    for (const [sourceRelative, destination, manifestRelative] of mappings) {
      copyTree(path.join(legacyRoot, sourceRelative), destination, legacyRoot, `${legacyRoot}/${manifestRelative}`);
    }
    const marker = path.join(legacyRoot, 'training', '.cell-yolo-ready');
    if (fs.existsSync(marker)) copyChangedFile(marker, RUNTIME_MARKER, legacyRoot, `${legacyRoot}/training/.cell-yolo-ready`);
    const stateFile = path.join(legacyRoot, 'training', '.cell-yolo-job-state.json');
    if (fs.existsSync(stateFile)) copyChangedFile(stateFile, JOB_STATE_FILE, legacyRoot, `${legacyRoot}/training/.cell-yolo-job-state.json`);
  }
  const datasetYaml = path.join(CELL_YOLO_DIR, 'dataset', 'dataset.yaml');
  if (fs.existsSync(datasetYaml)) {
    const yaml = fs.readFileSync(datasetYaml, 'utf8');
    const normalized = yaml.replace(/^path:\s*.*$/m, `path: ${path.join(CELL_YOLO_DIR, 'dataset').replace(/\\/g, '/')}`);
    if (normalized !== yaml) fs.writeFileSync(datasetYaml, normalized, 'utf8');
  }
  fs.writeFileSync(MIGRATION_FILE, JSON.stringify(migrationState, null, 2));
  return { dataRoot: DATA_ROOT, scannedReleases: releaseRoots.length, importedFiles: importedFileCount };
}
