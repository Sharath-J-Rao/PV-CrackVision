import path from 'node:path';
import { fileURLToPath } from 'node:url';

/** Vite 8 requires Node ^20.19.0 or >=22.12.0. */
export function isSupportedNodeVersion(version) {
  const parts = String(version).split('.').map((part) => Number.parseInt(part, 10));
  if (parts.length < 2 || parts.some((part) => !Number.isInteger(part) || part < 0)) return false;
  const [major, minor] = parts;
  return (major === 20 && minor >= 19) || (major === 22 && minor >= 12) || major >= 23;
}

const invokedPath = process.argv[1] ? path.resolve(process.argv[1]) : '';
if (invokedPath === fileURLToPath(import.meta.url)) {
  if (!isSupportedNodeVersion(process.versions.node)) {
    console.error(`PV CrackVision requires Node.js 20.19+ or 22.12+. Found ${process.versions.node}.`);
    process.exitCode = 1;
  }
}
