// Single-command release version setter. package.json is the canonical version source.
// Usage: node scripts/set-release-version.mjs 1.4.14.3 ["changelog headline"]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const input = process.argv[2];
if (!input || !/^\d+\.\d+\.\d+(?:\.\d+)?$/.test(input)) {
  console.error('Usage: node scripts/set-release-version.mjs <major.minor.patch[.revision]> ["changelog headline"]');
  process.exit(2);
}
const semver = input.replace(/\.(\d+)$/, '-$1').replace(/^(\d+\.\d+\.\d+)$/, '$1');
const label = semver.replace(/-(\d+)$/, '.$1');
const read = (file) => fs.readFileSync(path.join(root, file), 'utf8');
const write = (file, value) => fs.writeFileSync(path.join(root, file), value);

const pkg = JSON.parse(read('package.json'));
const oldSemver = pkg.version;
const oldLabel = oldSemver.replace(/-(\d+)$/, '.$1');
pkg.version = semver;
write('package.json', `${JSON.stringify(pkg, null, 2)}\n`);

const lock = JSON.parse(read('package-lock.json'));
lock.version = semver;
if (lock.packages?.['']) lock.packages[''].version = semver;
write('package-lock.json', `${JSON.stringify(lock, null, 2)}\n`);
const metadata = JSON.parse(read('metadata.json'));
metadata.version = semver;
write('metadata.json', `${JSON.stringify(metadata, null, 2)}\n`);

// The Windows workflow and Publish_Desktop.ps1 read the version from package.json at run time, so they need no edits.

const changelogPath = 'CHANGELOG.md';
const changelog = read(changelogPath);
if (oldLabel !== label && !changelog.trimStart().startsWith(`## v${label}`)) {
  const note = process.argv[3] || 'release (describe the changes here)';
  write(changelogPath, `## v${label} - ${note}\n\n${changelog}`);
}

// Only the "Current release" banner is rewritten; historical mentions (e.g. "introduced in v1.4.17.4") stay as written.
const readme = read('README.md').replace(/\*\*Current release: v[\d.]+\.\*\*/, `**Current release: v${label}.**`);
write('README.md', readme);

// VALIDATION_v<label>.md files are per-release test records and are written by hand after verification, so they are never renamed here.
console.log(`Release version synchronized: ${label} (package semver ${semver}).`);
console.log('Canonical source: package.json -> version. Run npm run release:set -- <new.version> for the next release.');
