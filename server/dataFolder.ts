/** Data-folder management: describe, browse, validate, copy and re-point the persistent data root. */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { APP_ROOT, DATA_ROOT, DATA_ROOT_SOURCE, DEFAULT_DATA_ROOT, RUNTIME_DIR } from './dataPaths.ts';
import { CONFIG_FILE, readAppConfig, resolveDataRootFromPick, writeAppConfig } from './appConfig.ts';

const SAME = (a: string, b: string) => path.resolve(a).toLowerCase() === path.resolve(b).toLowerCase();
const INSIDE = (child: string, parent: string) => { const rel = path.relative(path.resolve(parent), path.resolve(child)); return !!rel && !rel.startsWith('..') && !path.isAbsolute(rel); };

/** v1.4.11: asynchronous walk that never blocks the event loop; same 150,000-file truncation limit as before. */
export async function dirSize(dir: string, fileLimit = 150000): Promise<{ bytes: number; truncated: boolean }> {
  let bytes = 0, files = 0, truncated = false;
  const walk = async (current: string): Promise<void> => {
    if (truncated) return;
    let entries: fs.Dirent[];
    try { entries = await fs.promises.readdir(current, { withFileTypes: true }); } catch { return; }
    for (const entry of entries) {
      if (truncated) return;
      if (entry.isSymbolicLink()) continue;
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) await walk(full);
      else if (entry.isFile()) {
        try { bytes += (await fs.promises.stat(full)).size; } catch { /* vanished */ }
        if (++files >= fileLimit) { truncated = true; return; }
      }
    }
  };
  await walk(dir);
  return { bytes, truncated };
}

export function isWritableDir(dir: string): boolean {
  try {
    fs.mkdirSync(dir, { recursive: true });
    const probe = path.join(dir, `.write-test-${process.pid}`);
    fs.writeFileSync(probe, 'ok'); fs.unlinkSync(probe);
    return true;
  } catch { return false; }
}

export function describeDataFolder() {
  const configured = readAppConfig().dataRoot;
  const pendingRoot = DATA_ROOT_SOURCE !== 'env' && configured && !SAME(configured, DATA_ROOT) ? path.resolve(configured) : undefined;
  return {
    current: DATA_ROOT,
    source: DATA_ROOT_SOURCE,                 // env | config | default
    isDefault: DATA_ROOT_SOURCE === 'default',
    defaultRoot: DEFAULT_DATA_ROOT,
    configFile: CONFIG_FILE,
    envOverride: DATA_ROOT_SOURCE === 'env',
    pendingRoot,                              // chosen but not yet active: restart required
    restartRequired: !!pendingRoot,
    writable: isWritableDir(DATA_ROOT),
    exists: fs.existsSync(DATA_ROOT),
  };
}

const USAGE_TTL_MS = 60_000;
let usageCache: { at: number; value: { training: number; models: number; runtime: number; truncated: boolean } } | undefined;
let usageInFlight: Promise<NonNullable<typeof usageCache>['value']> | undefined;

/** Cached for 60 s and de-duplicated, so repeatedly opening Settings never starts parallel directory walks. */
export function dataFolderUsage(force = false) {
  if (!force && usageCache && Date.now() - usageCache.at < USAGE_TTL_MS) return Promise.resolve(usageCache.value);
  usageInFlight ??= (async () => {
    try {
      const [training, models, runtime] = await Promise.all([dirSize(path.join(DATA_ROOT, 'training')), dirSize(path.join(DATA_ROOT, 'models')), dirSize(RUNTIME_DIR)]);
      const value = { training: training.bytes, models: models.bytes, runtime: runtime.bytes, truncated: training.truncated || models.truncated || runtime.truncated };
      usageCache = { at: Date.now(), value };
      return value;
    } finally { usageInFlight = undefined; }
  })();
  return usageInFlight;
}

/** Lists sub-folders for the in-app folder picker (loopback-only API). */
export function browseFolders(requested?: string) {
  let current = requested && requested.trim() ? path.resolve(requested.trim()) : os.homedir();
  while (!fs.existsSync(current) && path.dirname(current) !== current) current = path.dirname(current);
  let dirs: Array<{ name: string; path: string }> = [];
  try {
    dirs = fs.readdirSync(current, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.') && !/^(\$RECYCLE\.BIN|System Volume Information|node_modules)$/i.test(entry.name))
      .map((entry) => ({ name: entry.name, path: path.join(current, entry.name) }))
      .sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
  } catch { /* unreadable folder: show empty list */ }
  const parent = path.dirname(current);
  const roots: Array<{ name: string; path: string }> = [{ name: 'Home', path: os.homedir() }];
  if (process.platform === 'win32') for (const letter of 'CDEFGHIJKLMNOPQRSTUVWXYZ') { const drive = `${letter}:\\`; if (fs.existsSync(drive)) roots.push({ name: drive, path: drive }); }
  else roots.push({ name: '/', path: '/' });
  return { path: current, parent: parent === current ? null : parent, dirs, roots, resolvedDataRoot: resolveDataRootFromPick(current) };
}

export type SetDataFolderOptions = { copyExisting: boolean; includeRuntime: boolean };

/** Validates the pick, optionally copies the existing data (non-destructive), then stores it in the per-user config. */
export async function setDataFolder(picked: string, options: SetDataFolderOptions) {
  if (DATA_ROOT_SOURCE === 'env') throw new Error('The data folder is fixed by the PV_CRACKVISION_DATA_DIR environment variable. Remove the variable to choose a folder here.');
  if (!picked || !path.isAbsolute(picked)) throw new Error('Enter a full folder path (for example D:\\PVCrackVisionData).');
  const target = resolveDataRootFromPick(picked);
  if (path.dirname(target) === target) throw new Error('Choose a folder, not a drive root.');
  if (INSIDE(target, APP_ROOT)) throw new Error('Choose a folder outside the program folder; it would be replaced on upgrade.');
  if (!SAME(target, DATA_ROOT) && INSIDE(target, DATA_ROOT)) throw new Error('The new folder cannot be inside the current data folder.');
  if (!SAME(target, DATA_ROOT) && INSIDE(DATA_ROOT, target)) throw new Error('The new folder cannot contain the current data folder. Pick a sibling folder instead.');
  if (!isWritableDir(target)) throw new Error('That folder is not writable. Pick another location.');
  let copied = 0;
  if (options.copyExisting && !SAME(target, DATA_ROOT) && fs.existsSync(DATA_ROOT)) {
    await fs.promises.cp(DATA_ROOT, target, {
      recursive: true, force: false, errorOnExist: false,
      filter: (source) => {
        const rel = path.relative(DATA_ROOT, source);
        if (!rel) return true;
        const top = rel.split(path.sep)[0];
        if (!options.includeRuntime && top === 'runtime') return false;      // Python env is not relocatable; one-click rebuild
        if (path.basename(source) === '.cell-yolo-job-state.json') return false; // contains absolute paths of the old root
        try { if (fs.lstatSync(source).isSymbolicLink()) return false; } catch { return false; }
        copied++;
        return true;
      },
    });
  }
  writeAppConfig({ dataRoot: target });
  return { dataRoot: target, copiedEntries: copied, restartRequired: !SAME(target, DATA_ROOT) };
}

export function resetDataFolderToDefault() {
  if (DATA_ROOT_SOURCE === 'env') throw new Error('The data folder is fixed by PV_CRACKVISION_DATA_DIR.');
  writeAppConfig({ dataRoot: undefined });
  return { dataRoot: DEFAULT_DATA_ROOT, restartRequired: !SAME(DEFAULT_DATA_ROOT, DATA_ROOT) };
}
