// v1.4.11: fails when release identities disagree. Usage: node scripts/check-release-version.mjs [root] [expectedVersion]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export function checkReleaseVersion(root, expected) {
  const read = (p) => fs.readFileSync(path.join(root, p), 'utf8');
  const errors = [];
  const version = JSON.parse(read('package.json')).version;
  // package.json must be valid semver ("1.4.11-2"); the human release label is four-part ("1.4.11.2").
  const label = version.replace(/-(\d+)$/, '.$1');
  if (expected && expected !== version && expected !== label) errors.push(`package.json is ${version} (${label}), expected ${expected}`);
  const lock = JSON.parse(read('package-lock.json'));
  if (lock.version !== version) errors.push(`package-lock.json version ${lock.version} != ${version}`);
  if (lock.packages?.['']?.version !== version) errors.push(`package-lock.json root entry ${lock.packages?.['']?.version} != ${version}`);
  if (JSON.parse(read('metadata.json')).version !== version) errors.push('metadata.json version differs');
  const workflow = read('.github/workflows/windows-desktop.yml');
  if (/release\/v\d+\.\d+\.\d+(?:\.\d+)?-desktop/.test(workflow) || /PV-CrackVision-v\d+\.\d+\.\d+/.test(workflow)) errors.push('workflow hard-codes a version; it must read it from package.json');
  const stale = fs.readdirSync(root).filter((name) => /^Publish_v.*_Desktop\.ps1$/.test(name));
  if (stale.length) errors.push(`version-named publish script(s) found (use Publish_Desktop.ps1): ${stale.join(', ')}`);
  if (!fs.existsSync(path.join(root, 'Publish_Desktop.ps1'))) errors.push('Publish_Desktop.ps1 is missing');
  if (!new RegExp(`\\*\\*Current release: v${label.replace(/\./g, '\\.')}\\.\\*\\*`).test(read('README.md'))) errors.push(`README.md "Current release" is not v${label}`);
  if (!read('CHANGELOG.md').trimStart().startsWith(`## v${label}`)) errors.push(`CHANGELOG.md does not start with v${label}`);
  return { version, errors };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const root = path.resolve(process.argv[2] || path.join(path.dirname(fileURLToPath(import.meta.url)), '..'));
  const { version, errors } = checkReleaseVersion(root, process.argv[3]);
  if (errors.length) { console.error(`Release check FAILED for ${version}:\n - ${errors.join('\n - ')}`); process.exit(1); }
  console.log(`Release identity consistent: v${version}`);
}
