// v1.4.13.1: manage resources/models/manifest.json.
//   node scripts/model-manifest.mjs generate [--model-version <label>]   write manifest.json from the files in resources/models
//   node scripts/model-manifest.mjs check [--require]                   verify them (--require: fail when a model is missing)
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dir = path.join(root, 'resources', 'models');
const FILES = ['cell.onnx', 'cellCrackModel.user.json'];
const pkgVersion = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')).version;
const digits = (v) => (String(v).match(/\d+/g) ?? []).map(Number);
const cmp = (a, b) => { const x = digits(a), y = digits(b); for (let i = 0; i < Math.max(x.length, y.length); i++) { const d = (x[i] ?? 0) - (y[i] ?? 0); if (d) return Math.sign(d); } return 0; };
const sha = (f) => crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');

export function generate(modelVersion) {
  const files = {};
  for (const name of FILES) {
    const f = path.join(dir, name);
    if (!fs.existsSync(f)) { console.warn(`  skipped ${name} (not found)`); continue; }
    files[name] = { sha256: sha(f), size: fs.statSync(f).size };
    if (name.endsWith('.json')) files[name].schema = JSON.parse(fs.readFileSync(f, 'utf8')).schema;
  }
  const manifest = { modelVersion: modelVersion || new Date().toISOString().slice(0, 10), minAppVersion: pkgVersion, generatedAt: new Date().toISOString(), files };
  fs.writeFileSync(path.join(dir, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  return manifest;
}

export function check({ require = false } = {}) {
  const errors = [], warnings = [];
  let manifest = null;
  try { manifest = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json'), 'utf8')); } catch { /* handled below */ }
  const present = FILES.filter((n) => fs.existsSync(path.join(dir, n)));
  if (!present.length && !require) return { errors, warnings: ['No bundled models yet (resources/models is empty). The EXE would use classical detection only.'] };
  for (const name of FILES) {
    const f = path.join(dir, name);
    if (!fs.existsSync(f)) { (require ? errors : warnings).push(`${name} is not in resources/models.`); continue; }
    const entry = manifest?.files?.[name];
    if (!entry) { errors.push(`${name}: no manifest entry. Run "npm run models:manifest".`); continue; }
    if (fs.statSync(f).size !== entry.size) errors.push(`${name}: size differs from manifest.`);
    else if (sha(f) !== entry.sha256) errors.push(`${name}: checksum differs from manifest.`);
    if (name === 'cell.onnx' && fs.readFileSync(f).subarray(0, 1)[0] !== 0x08) errors.push('cell.onnx is not an ONNX file.');
    if (name.endsWith('.json')) { try { if (!JSON.parse(fs.readFileSync(f, 'utf8')).schema) errors.push(`${name}: no schema field.`); } catch { errors.push(`${name}: not valid JSON.`); } }
  }
  if (manifest?.minAppVersion && cmp(pkgVersion, manifest.minAppVersion) < 0) errors.push(`Models need app ${manifest.minAppVersion} or newer; this is ${pkgVersion}.`);
  if (manifest?.maxAppVersion && cmp(pkgVersion, manifest.maxAppVersion) > 0) errors.push(`Models were built for app ${manifest.maxAppVersion} or older; this is ${pkgVersion}.`);
  return { errors, warnings };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [cmd, ...rest] = process.argv.slice(2);
  fs.mkdirSync(dir, { recursive: true });
  if (cmd === 'generate') {
    const i = rest.indexOf('--model-version');
    const m = generate(i >= 0 ? rest[i + 1] : undefined);
    console.log(`manifest.json written (model version ${m.modelVersion}, ${Object.keys(m.files).length} file(s)).`);
  } else if (cmd === 'check') {
    const { errors, warnings } = check({ require: rest.includes('--require') });
    warnings.forEach((w) => console.warn(`WARNING: ${w}`));
    if (errors.length) { console.error(`Bundled model check FAILED:\n - ${errors.join('\n - ')}`); process.exit(1); }
    console.log('Bundled model check passed.');
  } else { console.error('Usage: node scripts/model-manifest.mjs <generate|check> [--require] [--model-version <label>]'); process.exit(2); }
}
