// Single-command release version setter. package.json is the canonical version source.
// Usage: node scripts/set-release-version.mjs 1.4.14.3
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const input = process.argv[2];
if (!input || !/^\d+\.\d+\.\d+(?:\.\d+)?$/.test(input)) {
  console.error('Usage: node scripts/set-release-version.mjs <major.minor.patch[.revision]>');
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

const workflowPath = '.github/workflows/windows-desktop.yml';
let workflow = read(workflowPath)
  .replaceAll(`release/v${oldLabel}-desktop`, `release/v${label}-desktop`)
  .replaceAll(`PV-CrackVision-v${oldLabel}-Windows-x64`, `PV-CrackVision-v${label}-Windows-x64`);
write(workflowPath, workflow);

const oldPublisher = `Publish_v${oldLabel}_Desktop.ps1`;
const newPublisher = `Publish_v${label}_Desktop.ps1`;
let publisher = read(oldPublisher)
  .replaceAll(oldLabel, label)
  .replaceAll(oldSemver, semver);
write(newPublisher, publisher);
if (oldPublisher !== newPublisher) fs.rmSync(path.join(root, oldPublisher));

const changelog = read('CHANGELOG.md');
const lines = changelog.split(/\r?\n/);
if (lines[0]?.startsWith('## v')) lines[0] = `## v${label} - release version synchronized from package.json`;
write('CHANGELOG.md', lines.join('\n'));

// Keep the current-release references and versioned backup fixture aligned too.
for (const file of ['README.md', 'server/cellTrainingDataBundle.test.mjs']) {
  const contents = read(file).replaceAll(oldLabel, label).replaceAll(oldSemver, semver);
  write(file, contents);
}

const oldValidation = `VALIDATION_v${oldLabel}.md`;
const newValidation = `VALIDATION_v${label}.md`;
if (fs.existsSync(path.join(root, oldValidation)) && oldValidation !== newValidation) {
  const body = read(oldValidation).replaceAll(oldLabel, label).replaceAll(oldSemver, semver);
  write(newValidation, body);
  fs.rmSync(path.join(root, oldValidation));
}
console.log(`Release version synchronized: ${label} (package semver ${semver}).`);
console.log('Canonical source: package.json -> version. Run npm run release:set -- <new.version> for the next release.');
