/**
 * Per-user application config that lives OUTSIDE every release/install folder, so the chosen data
 * folder survives version upgrades, re-extracted ZIPs and a future packaged .exe.
 *
 *   Windows : %APPDATA%\PV CrackVision\config.json
 *   macOS   : ~/Library/Application Support/PV CrackVision/config.json
 *   Linux   : $XDG_CONFIG_HOME or ~/.config/PV CrackVision/config.json
 *
 * Override the folder with PV_CRACKVISION_CONFIG_DIR (portable installs, tests).
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

export const APP_NAME = 'PV CrackVision';
/** Folder name created inside a user-picked parent when it is not already a data folder. */
export const DATA_FOLDER_NAME = 'PV CrackVision Data';
const DATA_MARKERS = ['training', 'models', 'runtime'];

export function userConfigDir(env: NodeJS.ProcessEnv = process.env, platform: NodeJS.Platform = process.platform, home: string = os.homedir()): string {
  const override = env.PV_CRACKVISION_CONFIG_DIR?.trim();
  if (override) return path.resolve(override);
  if (platform === 'win32') return path.join(env.APPDATA?.trim() || path.join(home, 'AppData', 'Roaming'), APP_NAME);
  if (platform === 'darwin') return path.join(home, 'Library', 'Application Support', APP_NAME);
  return path.join(env.XDG_CONFIG_HOME?.trim() || path.join(home, '.config'), APP_NAME);
}

export const CONFIG_FILE = path.join(userConfigDir(), 'config.json');

export type AppConfig = { dataRoot?: string; updatedAt?: string };

export function readAppConfig(file: string = CONFIG_FILE): AppConfig {
  try {
    const parsed = JSON.parse(fs.readFileSync(file, 'utf8'));
    return parsed && typeof parsed === 'object' ? parsed as AppConfig : {};
  } catch { return {}; }
}

export function writeAppConfig(patch: Partial<AppConfig>, file: string = CONFIG_FILE): AppConfig {
  const next: AppConfig = { ...readAppConfig(file), ...patch, updatedAt: new Date().toISOString() };
  for (const key of Object.keys(next) as Array<keyof AppConfig>) if (next[key] === undefined) delete next[key];
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.tmp`;
  fs.writeFileSync(temporary, JSON.stringify(next, null, 2), 'utf8');
  fs.renameSync(temporary, file);
  return next;
}

/** The data folder the user explicitly chose, if it is a usable absolute path. */
export function configuredDataRoot(file: string = CONFIG_FILE): string | undefined {
  const value = readAppConfig(file).dataRoot;
  return typeof value === 'string' && value.trim() && path.isAbsolute(value.trim()) ? path.resolve(value.trim()) : undefined;
}

/** True for a packaged build (pkg/Electron/explicit flag): there is no writable "Working Folder" next to the program. */
export function isPackagedApp(env: NodeJS.ProcessEnv = process.env): boolean {
  return env.PV_CRACKVISION_PACKAGED === '1' || Boolean((process as any).pkg) || Boolean(process.versions && (process.versions as any).electron);
}

/**
 * Turns whatever folder the user picked into the real data root. A folder that already is a data folder
 * (name ends in "data" or it holds our sub-folders) is used as-is; any other folder gets a dedicated sub-folder
 * so user files are never mixed with ours.
 */
export function resolveDataRootFromPick(picked: string): string {
  const absolute = path.resolve(picked);
  const looksLikeData = /data$/i.test(path.basename(absolute))
    || DATA_MARKERS.some((marker) => { try { return fs.statSync(path.join(absolute, marker)).isDirectory(); } catch { return false; } });
  return looksLikeData ? absolute : path.join(absolute, DATA_FOLDER_NAME);
}
