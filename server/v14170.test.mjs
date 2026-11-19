import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// Isolate the data folder BEFORE any module that resolves paths is loaded (static imports would run first).
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pv-anom-'));
process.env.PV_CRACKVISION_DATA_DIR = dir;
const { DEFAULT_TRAINING_SETTINGS, normalizeTrainingSettings } = await import('./trainingSettings.ts');

test('v1.4.17: anomaly detector is the default crack engine; the archived classifier stays selectable', () => {
  assert.equal(normalizeTrainingSettings({}).crackDetection.engine, 'anomaly');
  assert.equal(normalizeTrainingSettings({ crackDetection: { engine: 'classifier' } }).crackDetection.engine, 'classifier');
  assert.equal(normalizeTrainingSettings({ crackDetection: { engine: 'bogus' } }).crackDetection.engine, 'anomaly');
  assert.equal(normalizeTrainingSettings({ crackDetection: { decisionThreshold: 0.3 } }).crackDetection.engine, 'anomaly'); // files saved before v1.4.17 have no engine
});

test('v1.4.17: anomaly settings default, round-trip and clamp', () => {
  const d = DEFAULT_TRAINING_SETTINGS.crackAnomaly;
  assert.deepEqual(normalizeTrainingSettings({}).crackAnomaly, d);
  assert.equal(d.imgsz % 32, 0);
  const s = normalizeTrainingSettings({ crackAnomaly: { imgsz: 250, memoryBankSize: 10, topKPatches: 999, valFraction: 0.9, cleanQuantile: 2, backbone: 'x.pt', seed: -4 } }).crackAnomaly;
  assert.equal(s.imgsz, 256); // snapped to a multiple of 32
  assert.equal(s.memoryBankSize, 1000);
  assert.equal(s.topKPatches, 50);
  assert.equal(s.valFraction, 0.5);
  assert.equal(s.cleanQuantile, 0.9999);
  assert.equal(s.backbone, d.backbone);
  assert.equal(s.seed, 0);
  assert.equal(normalizeTrainingSettings({ crackAnomaly: { imgsz: 'abc' } }).crackAnomaly.imgsz, d.imgsz);
  assert.equal(normalizeTrainingSettings({ crackAnomaly: { backbone: 'yolo11s-cls.pt' } }).crackAnomaly.backbone, 'yolo11s-cls.pt');
});

test('v1.4.17: anomaly model files are separate from the archived classifier and validated before use', async () => {
  const { activeClassifierFiles, loadCellClassifier, getCellClassifierWarning } = await import('./cellClassifier.ts');
  const files = activeClassifierFiles('anomaly');
  assert.match(path.basename(files.onnx), /^cellAnomaly\.onnx$/);
  assert.match(path.basename(files.meta), /^cellAnomaly\.json$/);
  assert.notEqual(files.onnx, activeClassifierFiles('classifier').onnx);
  assert.equal(await loadCellClassifier('anomaly'), null); // nothing trained yet
  fs.mkdirSync(path.dirname(files.onnx), { recursive: true });
  fs.writeFileSync(files.onnx, Buffer.from([0x08, 1, 2, 3]));
  fs.writeFileSync(files.meta, JSON.stringify({ schema: 'cell-classifier-v1', imgsz: 224, crackedIndex: 1, threshold: 0.5, cellBuffer: 0.03 })); // wrong schema for this engine
  assert.equal(await loadCellClassifier('anomaly'), null);
  assert.match(getCellClassifierWarning('anomaly') ?? '', /missing or invalid/);
  fs.writeFileSync(files.meta, JSON.stringify({ schema: 'cell-anomaly-v1', imgsz: 224, crackedIndex: 1, threshold: 0.5, cellBuffer: 0.03, batchSize: 0 })); // bad batch size
  assert.equal(await loadCellClassifier('anomaly'), null);
});

test('v1.4.17: job runner knows the anomaly mode', async () => {
  const { isClassifierMode } = await import('./classifierTraining.ts');
  for (const mode of ['anom', 'cls', 'det']) assert.equal(isClassifierMode(mode), true);
  assert.equal(isClassifierMode('anomaly'), false);
});

test('v1.4.17.1: a pre-engine decisionThreshold override resets (classifier scale); an explicit engine keeps it', () => {
  assert.equal(normalizeTrainingSettings({ crackDetection: { decisionThreshold: 0.3 } }).crackDetection.decisionThreshold, 0);
  assert.equal(normalizeTrainingSettings({ crackDetection: { engine: 'classifier', decisionThreshold: 0.3 } }).crackDetection.decisionThreshold, 0.3);
  assert.equal(normalizeTrainingSettings({ crackDetection: { engine: 'anomaly', decisionThreshold: 0.6 } }).crackDetection.decisionThreshold, 0.6);
});
