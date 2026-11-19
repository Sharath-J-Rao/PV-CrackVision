import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import os from 'node:os';

function usesWindowsPath(value) { return /^[A-Za-z]:[\\/]/.test(String(value)) || /^\\\\/.test(String(value)); }

export function findWorkingFolder(appRoot) {
  const pathApi = usesWindowsPath(appRoot) ? path.win32 : path;
  let current = pathApi.resolve(appRoot);
  while (true) {
    if (pathApi.basename(current).toLowerCase() === 'working folder') return current;
    const parent = pathApi.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return pathApi.resolve(appRoot, '..');
}

export function resolveSharedDataRoot(appRoot, environment = process.env) {
  const override = typeof environment.PV_CRACKVISION_DATA_DIR === 'string'
    ? environment.PV_CRACKVISION_DATA_DIR.trim()
    : '';
  const pathApi = usesWindowsPath(override || appRoot) ? path.win32 : path;
  return pathApi.resolve(override || pathApi.join(findWorkingFolder(appRoot), 'data'));
}

export function discoverReleaseRoots(workingFolder, currentRoot) {
  const candidates = new Set([path.resolve(currentRoot)]);
  const releaseName = /^PV[_-]?CrackVision/i;
  const scan = (directory, depth) => {
    let entries;
    try { entries = fs.readdirSync(directory, { withFileTypes: true }); } catch { return; }
    for (const entry of entries) {
      if (!entry.isDirectory() || !releaseName.test(entry.name)) continue;
      const candidate = path.join(directory, entry.name);
      if (fs.existsSync(path.join(candidate, 'server.ts')) && fs.existsSync(path.join(candidate, 'server')) && fs.existsSync(path.join(candidate, 'training'))) {
        candidates.add(candidate);
      } else if (depth < 4) scan(candidate, depth + 1);
    }
  };
  scan(workingFolder, 0);
  return [...candidates];
}

export function hasCellModelInReleaseFolders(workingFolder, currentRoot) {
  return discoverReleaseRoots(workingFolder, currentRoot).some((releaseRoot) => fs.existsSync(path.join(releaseRoot, 'server', 'models', 'cell.onnx')));
}

/** Mirrors server/appConfig.ts so start.bat/start.sh see the same data folder as the server. */
export function userConfigDir(environment = process.env, platform = process.platform, home = os.homedir()) {
  const override = typeof environment.PV_CRACKVISION_CONFIG_DIR === 'string' ? environment.PV_CRACKVISION_CONFIG_DIR.trim() : '';
  if (override) return path.resolve(override);
  if (platform === 'win32') return path.join((environment.APPDATA || '').trim() || path.join(home, 'AppData', 'Roaming'), 'PV CrackVision');
  if (platform === 'darwin') return path.join(home, 'Library', 'Application Support', 'PV CrackVision');
  return path.join((environment.XDG_CONFIG_HOME || '').trim() || path.join(home, '.config'), 'PV CrackVision');
}

export function resolveActiveDataRoot(appRoot, environment = process.env) {
  if (typeof environment.PV_CRACKVISION_DATA_DIR === 'string' && environment.PV_CRACKVISION_DATA_DIR.trim()) return resolveSharedDataRoot(appRoot, environment);
  try {
    const configured = JSON.parse(fs.readFileSync(path.join(userConfigDir(environment), 'config.json'), 'utf8')).dataRoot;
    if (typeof configured === 'string' && path.isAbsolute(configured.trim())) return path.resolve(configured.trim());
  } catch { /* no config yet: fall back to the default location */ }
  return resolveSharedDataRoot(appRoot, environment);
}

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const invokedPath = process.argv[1] ? path.resolve(process.argv[1]) : '';
if (invokedPath === fileURLToPath(import.meta.url)) {
  if (process.argv[2] === '--has-cell-model') {
    process.stdout.write(`${hasCellModelInReleaseFolders(findWorkingFolder(appRoot), appRoot)}\n`);
  } else {
    process.stdout.write(`${resolveActiveDataRoot(appRoot)}\n`);
  }
}
