// Generates THIRD_PARTY_NOTICES.md from package-lock.json (production dependencies only; works offline, deterministic).
// Usage: node scripts/generate-third-party-notices.mjs [--check]   (--check exits 1 when the file is stale)
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export function buildNotices(lock) {
  const rows = [];
  for (const [key, value] of Object.entries(lock.packages ?? {})) {
    if (!key || value.dev) continue;
    const name = key.slice(key.lastIndexOf('node_modules/') + 'node_modules/'.length);
    rows.push({ name, version: value.version, license: value.license || 'UNKNOWN' });
  }
  rows.sort((a, b) => a.name.localeCompare(b.name) || a.version.localeCompare(b.version));
  const unique = [...new Map(rows.map((r) => [`${r.name}@${r.version}`, r])).values()];
  const byLicense = new Map();
  for (const r of unique) byLicense.set(r.license, (byLicense.get(r.license) ?? 0) + 1);
  const summary = [...byLicense.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([l, n]) => `| ${l} | ${n} |`);
  return [
    '# Third-party notices',
    '',
    'PV CrackVision bundles or loads the open-source components below. Each component remains under its own license; the full license text of every npm package ships in that package\'s folder inside the application bundle. This file is generated from `package-lock.json` by `npm run notices` and is verified by `npm run notices:check` (and in CI) so it cannot drift from the locked dependencies.',
    '',
    '## Summary by license',
    '',
    '| License | Packages |',
    '|---|---|',
    ...summary,
    '',
    '## Components bundled at packaging time (not in the npm lock)',
    '',
    '- **Electron** and **Chromium** (MIT and many third-party licenses). The installer includes Electron\'s `LICENSE` and `LICENSES.chromium.html`.',
    '- **sharp / libvips** native binaries (Apache-2.0, LGPL-3.0-or-later): distributed as separate dynamically loaded libraries under `app.asar.unpacked`, so the LGPL libraries can be replaced by the user.',
    '- **ONNX Runtime** (MIT).',
    '',
    '## Components installed on the user\'s machine on request (Python training environment)',
    '',
    'The optional "Train cell detector" feature sets up a local Python environment and runs `training/train_cell_yolo.py`, which imports **Ultralytics YOLO** (AGPL-3.0, with a separate commercial Enterprise license available from Ultralytics). It is downloaded by the user\'s machine at training time and is not part of this repository or the installer. **Legal review is required before commercial redistribution or hosted use** (see SECURITY.md, "Commercial-release gates").',
    '',
    '## npm production dependencies',
    '',
    '| Package | Version | License |',
    '|---|---|---|',
    ...unique.map((r) => `| ${r.name} | ${r.version} | ${r.license} |`),
    '',
  ].join('\n');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const target = path.join(root, 'THIRD_PARTY_NOTICES.md');
  const next = buildNotices(JSON.parse(fs.readFileSync(path.join(root, 'package-lock.json'), 'utf8')));
  if (process.argv.includes('--check')) {
    const current = fs.existsSync(target) ? fs.readFileSync(target, 'utf8') : '';
    if (current !== next) { console.error('THIRD_PARTY_NOTICES.md is missing or stale. Run: npm run notices'); process.exit(1); }
    console.log('Third-party notices are current.');
  } else {
    fs.writeFileSync(target, next);
    console.log(`Wrote THIRD_PARTY_NOTICES.md (${next.split('\n').length} lines).`);
  }
}
