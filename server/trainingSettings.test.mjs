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
  assert.equal(settings.crackClassifier.iterations, 1400);
  assert.equal(settings.crackClassifier.classBalance, true);
});

test('training settings clamp invalid values and reject unsupported image sizes and optimizers', () => {
  const settings = normalizeTrainingSettings({
    cellDetector: { epochs: 9999, patience: 9999, imageSize: 123, freeze: -8, batch: 0, optimizer: 'Bogus', lr0: -1 },
    crackClassifier: { iterations: 1, validationFraction: 0.99, thresholdMin: 0.7, thresholdMax: 0.2, thresholdStep: 0 },
  });
  assert.equal(settings.cellDetector.epochs, 500);
  assert.equal(settings.cellDetector.patience, 500);
  assert.equal(settings.cellDetector.imageSize, DEFAULT_TRAINING_SETTINGS.cellDetector.imageSize);
  assert.equal(settings.cellDetector.freeze, 0);
  assert.equal(settings.cellDetector.batch, 1);
  assert.equal(settings.cellDetector.optimizer, 'AdamW');
  assert.equal(settings.cellDetector.minimumMap50, 0.05);
  assert.equal(settings.cellDetector.lr0, 0.000001);
  assert.equal(settings.crackClassifier.iterations, 100);
  assert.equal(settings.crackClassifier.validationFraction, 0.4);
  assert.ok(settings.crackClassifier.thresholdMax > settings.crackClassifier.thresholdMin);
  assert.equal(settings.crackClassifier.thresholdStep, 0.005);
});
