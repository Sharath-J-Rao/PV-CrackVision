import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { resolveModel, checkBundledFile, compareVersions, resolveClassifierFiles } from './modelResolver.ts';

const onnx = () => Buffer.concat([Buffer.from([0x08, 0x07]), Buffer.alloc(4096, 1)]);
const crack = () => Buffer.from(JSON.stringify({ schema: 'cell-level-v2', pad: 'x'.repeat(2000) }));
function fixture({ manifest = {}, tamper = false } = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'pv-models-'));
  const user = path.join(root, 'user'), bundled = path.join(root, 'bundled');
  fs.mkdirSync(user); fs.mkdirSync(bundled);
  const files = { 'cell.onnx': onnx(), 'cellCrackModel.user.json': crack() };
  const entries = {};
  for (const [name, buf] of Object.entries(files)) {
    entries[name] = { sha256: crypto.createHash('sha256').update(buf).digest('hex'), size: buf.length };
    fs.writeFileSync(path.join(bundled, name), tamper && name === 'cell.onnx' ? Buffer.concat([buf.subarray(0, -1), Buffer.from([9])]) : buf);
  }
  fs.writeFileSync(path.join(bundled, 'manifest.json'), JSON.stringify({ modelVersion: 't', minAppVersion: '1.4.13-1', files: entries, ...manifest }));
  return { root, user, bundled };
}

test('versions compare numerically across - and . separators', () => {
  assert.equal(compareVersions('1.4.13-1', '1.4.13.1'), 0);
  assert.equal(compareVersions('1.4.13-2', '1.4.13.1'), 1);
  assert.equal(compareVersions('1.4.9', '1.4.13'), -1);
});

test('bundled model is used when the user has none', () => {
  const f = fixture();
  const r = resolveModel('cell.onnx', f.user, f.bundled, '1.4.13-1');
  assert.deepEqual(r.candidates.map((c) => c.source), ['bundled']);
  assert.equal(r.warnings.length, 0);
});

test('a valid user model wins and the bundled one remains the fallback', () => {
  const f = fixture();
  fs.writeFileSync(path.join(f.user, 'cell.onnx'), onnx());
  assert.deepEqual(resolveModel('cell.onnx', f.user, f.bundled, '1.4.13-1').candidates.map((c) => c.source), ['user', 'bundled']);
});

test('corrupt bundled model (checksum mismatch) is skipped with a warning, not thrown', () => {
  const f = fixture({ tamper: true });
  const r = resolveModel('cell.onnx', f.user, f.bundled, '1.4.13-1');
  assert.equal(r.candidates.length, 0);
  assert.match(r.warnings[0], /checksum/);
});

test('truncated bundled model and non-ONNX user file are rejected', () => {
  const f = fixture();
  fs.writeFileSync(path.join(f.bundled, 'cell.onnx'), onnx().subarray(0, 2000));
  fs.writeFileSync(path.join(f.user, 'cell.onnx'), Buffer.alloc(5000, 7));
  const r = resolveModel('cell.onnx', f.user, f.bundled, '1.4.13-1');
  assert.equal(r.candidates.length, 0);
  assert.equal(r.warnings.length, 2);
});

test('models built for a newer app are rejected as incompatible', () => {
  const f = fixture({ manifest: { minAppVersion: '1.5.0-1' } });
  const c = checkBundledFile('cell.onnx', f.bundled, '1.4.13-1');
  assert.equal(c.ok, false);
  assert.match(c.reason, /needs app/);
});

test('missing manifest or missing bundle never throws', () => {
  const f = fixture();
  fs.rmSync(path.join(f.bundled, 'manifest.json'));
  assert.equal(resolveModel('cell.onnx', f.user, f.bundled, '1.4.13-1').candidates.length, 0);
  assert.equal(resolveModel('cell.onnx', f.user, path.join(f.root, 'nope'), '1.4.13-1').warnings.length, 0);
});

test('crack classifier JSON resolves the same way', () => {
  const f = fixture();
  assert.deepEqual(resolveModel('cellCrackModel.user.json', f.user, f.bundled, '1.4.13-1').candidates.map((c) => c.source), ['bundled']);
});

test('classifier pair: user pair wins, else verified bundled pair, else none', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'pv-cls-'));
  const user = path.join(root, 'user'), bundled = path.join(root, 'bundled');
  fs.mkdirSync(user); fs.mkdirSync(bundled);
  const meta = Buffer.from(JSON.stringify({ schema: 'cell-classifier-v1' }));
  const entries = {};
  for (const [n, b] of [['cellClassifier.onnx', onnx()], ['cellClassifier.json', meta]]) {
    fs.writeFileSync(path.join(bundled, n), b);
    entries[n] = { sha256: crypto.createHash('sha256').update(b).digest('hex'), size: b.length };
  }
  fs.writeFileSync(path.join(bundled, 'manifest.json'), JSON.stringify({ minAppVersion: '1.4.13-1', files: entries }));
  assert.equal(resolveClassifierFiles(user, bundled, '1.4.15-0')?.source, 'bundled');
  fs.writeFileSync(path.join(user, 'cellClassifier.onnx'), onnx());
  assert.equal(resolveClassifierFiles(user, bundled, '1.4.15-0')?.source, 'bundled'); // half a user pair is ignored
  fs.writeFileSync(path.join(user, 'cellClassifier.json'), meta);
  assert.equal(resolveClassifierFiles(user, bundled, '1.4.15-0')?.source, 'user');
  fs.rmSync(path.join(bundled, 'cellClassifier.json'));
  fs.rmSync(user, { recursive: true });
  assert.equal(resolveClassifierFiles(user, bundled, '1.4.15-0'), null);
});
