import fs from 'node:fs';
import path from 'node:path';

/** Resolve a script passed to an external process (Python cannot read Electron's virtual app.asar paths). */
export function resolveExternalWorkerPath(relativePath: string, appRoot: string, electronResourcesPath?: string): string {
  const parts = relativePath.split(/[\\/]+/).filter(Boolean);
  if (electronResourcesPath) {
    const unpacked = path.join(electronResourcesPath, 'app.asar.unpacked', ...parts);
    if (fs.existsSync(unpacked)) return unpacked;
  }
  return path.join(appRoot, ...parts);
}
