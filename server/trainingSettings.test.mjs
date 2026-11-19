import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_TRAINING_SETTINGS, normalizeTrainingSettings } from './trainingSettings.ts';

test('training settings provide complete independent model defaults', () => {
  const settings = normalizeTrainingSettings({});
  assert.equal(settings.cellDetector.epochs, 200);
  assert.equal(settings.cellDetector.patience, 30);
  assert.equal(settings.cellDetector.freeze, 0);
  assert.equal(settings.cellDetector.optimizer, 'AdamW');
  assert.equal(settings.cellDetector.minimumMap50, 0.05);
  assert.equal(settings.crackClassifier.cellBuffer, 0.03);
  assert.equal(settings.workflow.autoDetectOnUpload, true); // v1.4.15.1: on by default
  assert.equal(normalizeTrainingSettings({ workflow: { autoDetectOnUpload: false } }).workflow.autoDetectOnUpload, false);
});

test('training settings clamp invalid values and reject unsupported image sizes and optimizers', () => {
  const settings = normalizeTrainingSettings({
    cellDetector: { epochs: 9999, patience: 9999, imageSize: 123, freeze: -8, batch: 0, optimizer: 'Bogus', lr0: -1 },
    crackClassifier: { cellBuffer: 5 },
  });
  assert.equal(settings.cellDetector.epochs, 500);
  assert.equal(settings.cellDetector.patience, 500);
  assert.equal(settings.cellDetector.imageSize, DEFAULT_TRAINING_SETTINGS.cellDetector.imageSize);
  assert.equal(settings.cellDetector.freeze, 0);
  assert.equal(settings.cellDetector.batch, 1);
  assert.equal(settings.cellDetector.optimizer, 'AdamW');
  assert.equal(settings.cellDetector.minimumMap50, 0.05);
  assert.equal(settings.cellDetector.lr0, 0.000001);
  assert.equal(settings.crackClassifier.cellBuffer, 0.1);
});
