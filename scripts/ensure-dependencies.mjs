/**
 * v1.4.12.7 hotfix: install missing npm packages automatically before the app starts.
 * Previously a missing/partial node_modules (interrupted first install, new dependency after an update) only produced
 * "Run npm install" and the app did not start. This checks every package declared in package.json and installs once if any is absent.
 */
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

/** Names declared in package.json (dependencies + devDependencies) that have no package.json under node_modules. */
export function findMissingDependencies(root) {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
  const names = [...Object.keys(manifest.dependencies ?? {}), ...Object.keys(manifest.devDependencies ?? {})];
  return names.filter((name) => !fs.existsSync(path.join(root, 'node_modules', ...name.split('/'), 'package.json')));
}

/** Runs `npm install` when something is missing. Returns true when everything is present afterwards. */
export function ensureDependencies(root, log = console.log) {
  let missing;
  try { missing = findMissingDependencies(root); }
  catch (error) { log(`Could not read package.json to check dependencies: ${error.message}`); return false; }
  if (missing.length === 0) return true;
  log(`Missing packages (${missing.join(', ')}). Installing automatically - this can take a few minutes...`);
  // shell:true only on Windows, where npm is npm.cmd and Node refuses to spawn .cmd files directly.
  const result = spawnSync('npm', ['install', '--no-audit', '--no-fund'], { cwd: root, stdio: 'inherit', shell: process.platform === 'win32' });
  if (result.error || result.status !== 0) {
    log(`Automatic install failed${result.error ? `: ${result.error.message}` : ` (exit ${result.status})`}. Check the internet connection, then run npm install manually.`);
    return false;
  }
  return findMissingDependencies(root).length === 0;
}

// Allow `node scripts/ensure-dependencies.mjs` as a standalone check.
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = ensureDependencies(path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')) ? 0 : 1;
}
