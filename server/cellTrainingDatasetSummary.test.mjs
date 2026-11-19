import test from 'node:test';
import assert from 'node:assert/strict';
import { buildCellTrainingDatasetSummary } from './cellTrainingDatasetSummary.ts';

test('dataset summary exposes validationBoxes from the validation split', () => {
  const summary = buildCellTrainingDatasetSummary({
    yamlPath: 'training/cell_yolo/dataset/dataset.yaml',
    images: 5,
    boxes: 120,
    trainImages: 4,
    validationImages: 1,
    trainBoxes: 96,
    validationBoxes: 24,
  });
  assert.equal(summary.validationBoxes, 24);
  assert.equal(summary.trainBoxes + summary.validationBoxes, summary.boxes);
  assert.equal('valBoxes' in summary, false);
});
