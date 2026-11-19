import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';

// Isolate the data folder BEFORE any module that resolves paths is loaded.
process.env.PV_CRACKVISION_DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'pv-bundled-anom-'));
const { resolveAnomalyFiles, bundledAnomalyFiles } = await import('./modelResolver.ts');

const APP = '1.4.17.6';
const onnx = () => Buffer.concat([Buffer.from([0x08]), Buffer.alloc(2048, 7)]);
const meta = () => JSON.stringify({ schema: 'cell-anomaly-v1', version: 't', imgsz: 224, crackedIndex: 1, threshold: 0.5, cellBuffer: 0.03 });
const sha = (b) => crypto.createHash('sha256').update(b).digest('hex');
const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), 'pv-anom-res-'));
function writePair(dir, { manifest = true, onnxBytes = onnx() } = {}) {
  fs.mkdirSync(dir, { recursive: true });
  const m = Buffer.from(meta());
  fs.writeFileSync(path.join(dir, 'cellAnomaly.onnx'), onnxBytes); fs.writeFileSync(path.join(dir, 'cellAnomaly.json'), m);
  if (manifest) fs.writeFileSync(path.join(dir, 'manifest.json'), JSON.stringify({ minAppVersion: '1.4.17.6', files: {
    'cellAnomaly.onnx': { sha256: sha(onnx()), size: onnx().length }, 'cellAnomaly.json': { sha256: sha(m), size: m.length, schema: 'cell-anomaly-v1' } } }));
}

test('v1.4.17.6: nothing user-trained and nothing bundled resolves to null', () => {
  assert.equal(resolveAnomalyFiles(tmp(), tmp(), APP), null);
});

test('v1.4.17.6: the verified bundled pair is used when the user has not trained one', () => {
  const bundled = tmp(); writePair(bundled);
  const r = resolveAnomalyFiles(tmp(), bundled, APP);
  assert.equal(r?.source, 'bundled');
  assert.equal(path.basename(r.onnx), 'cellAnomaly.onnx');
});

test('v1.4.17.6: a model trained by the user (also from an older version) wins over the bundled one', () => {
  const bundled = tmp(), user = tmp(); writePair(bundled); writePair(user, { manifest: false });
  const r = resolveAnomalyFiles(user, bundled, APP);
  assert.equal(r?.source, 'user');
  assert.ok(r.onnx.startsWith(user));
});

test('v1.4.17.6: a broken user pair is skipped (does not shadow the bundled one); a half pair is not a user pair', () => {
  const bundled = tmp(), user = tmp(); writePair(bundled); writePair(user, { manifest: false, onnxBytes: Buffer.from([1, 2, 3]) });
  assert.equal(resolveAnomalyFiles(user, bundled, APP)?.source, 'bundled');
  const half = tmp(); fs.writeFileSync(path.join(half, 'cellAnomaly.onnx'), onnx());
  assert.equal(resolveAnomalyFiles(half, bundled, APP)?.source, 'bundled');
});

test('v1.4.17.6: a corrupt, unlisted or too-new bundled pair is rejected', () => {
  const bad = tmp(); writePair(bad); fs.appendFileSync(path.join(bad, 'cellAnomaly.onnx'), 'x'); // size/checksum no longer match the manifest
  assert.equal(bundledAnomalyFiles(bad, APP), null);
  const noManifest = tmp(); writePair(noManifest, { manifest: false });
  assert.equal(bundledAnomalyFiles(noManifest, APP), null);
  const good = tmp(); writePair(good);
  assert.equal(bundledAnomalyFiles(good, '1.4.17.5'), null); // manifest needs app 1.4.17.6+
  assert.notEqual(bundledAnomalyFiles(good, APP), null);
});
