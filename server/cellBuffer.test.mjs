import test from 'node:test';
import assert from 'node:assert/strict';
import { bufferedCellCrop, DEFAULT_CELL_BUFFER } from './crackModel.ts';
import { normalizeTrainingSettings, DEFAULT_TRAINING_SETTINGS } from './trainingSettings.ts';

const box = { left: 100, top: 100, width: 100, height: 100 };

test('buffer 0 reproduces the exact cell box (old models score exactly as before)', () => {
  assert.deepEqual(bufferedCellCrop(1000, 1000, box, 0), { left: 100, top: 100, width: 100, height: 100 });
  assert.deepEqual(bufferedCellCrop(1000, 1000, box), { left: 100, top: 100, width: 100, height: 100 });
});

test('default buffer is minimal: 3% of the short side, at least 1 px, grown on every side', () => {
  assert.equal(DEFAULT_CELL_BUFFER, 0.03);
  assert.deepEqual(bufferedCellCrop(1000, 1000, box, 0.03), { left: 97, top: 97, width: 106, height: 106 });
  const small = bufferedCellCrop(1000, 1000, { left: 50, top: 50, width: 20, height: 20 }, 0.03); // 0.6 px -> 1 px minimum
  assert.deepEqual(small, { left: 49, top: 49, width: 22, height: 22 });
});

test('buffer is clamped at image borders', () => {
  assert.deepEqual(bufferedCellCrop(205, 200, { left: 2, top: 0, width: 200, height: 200 }, 0.03), { left: 0, top: 0, width: 205, height: 200 });
});

test('settings: cellBuffer defaults to 0.03 and is clamped to 0..0.1', () => {
  assert.equal(DEFAULT_TRAINING_SETTINGS.crackClassifier.cellBuffer, 0.03);
  assert.equal(normalizeTrainingSettings({ crackClassifier: { cellBuffer: 5 } }).crackClassifier.cellBuffer, 0.1);
  assert.equal(normalizeTrainingSettings({ crackClassifier: { cellBuffer: -1 } }).crackClassifier.cellBuffer, 0);
  assert.equal(normalizeTrainingSettings({ crackClassifier: { cellBuffer: 0 } }).crackClassifier.cellBuffer, 0);
});
