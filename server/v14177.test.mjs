import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

process.env.PV_CRACKVISION_DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'pv-recover-'));
const { readValidPair } = await import('./modelRecovery.ts');

const onnx = () => Buffer.concat([Buffer.from([0x08]), Buffer.alloc(2048, 7)]);
const meta = (extra = {}) => JSON.stringify({ schema: 'cell-anomaly-v1', version: 'old', imgsz: 224, crackedIndex: 1, threshold: 0.5, cellBuffer: 0.03, trainedAt: '2026-01-02T03:04:05', ...extra });
const dir = () => fs.mkdtempSync(path.join(os.tmpdir(), 'pv-pair-'));

test('v1.4.17.7: a complete valid pair is recognised with its trained time', () => {
  const d = dir(); fs.writeFileSync(path.join(d, 'cellAnomaly.onnx'), onnx()); fs.writeFileSync(path.join(d, 'cellAnomaly.json'), meta());
  const p = readValidPair(d, 'anomaly'); assert.equal(p?.version, 'old'); assert.equal(p?.trainedAt, Date.parse('2026-01-02T03:04:05'));
});
test('v1.4.17.7: half pairs, wrong schema and non-ONNX files are rejected', () => {
  const half = dir(); fs.writeFileSync(path.join(half, 'cellAnomaly.onnx'), onnx()); assert.equal(readValidPair(half, 'anomaly'), null);
  const wrong = dir(); fs.writeFileSync(path.join(wrong, 'cellAnomaly.onnx'), onnx()); fs.writeFileSync(path.join(wrong, 'cellAnomaly.json'), meta({ schema: 'cell-classifier-v1' })); assert.equal(readValidPair(wrong, 'anomaly'), null);
  const bad = dir(); fs.writeFileSync(path.join(bad, 'cellAnomaly.onnx'), Buffer.from([1, 2, 3])); fs.writeFileSync(path.join(bad, 'cellAnomaly.json'), meta()); assert.equal(readValidPair(bad, 'anomaly'), null);
});
