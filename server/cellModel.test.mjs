import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CELL_FEATURE_NAMES,
  CELL_WINDOW_MARGIN,
  CELL_WINDOW_SIZE,
  cellFeatureVector,
  cellCropFeatures,
} from './cellFeatures.ts';

test('cell feature vector is fixed-width and has no annotation-dependent context', () => {
  const raw = CELL_FEATURE_NAMES.map((_, index) => index + 1);
  const vector = cellFeatureVector(raw);
  assert.equal(vector.length, CELL_FEATURE_NAMES.length * 2);
  assert.deepEqual(vector.slice(0, raw.length), raw);
  assert.deepEqual(vector.slice(raw.length), new Array(raw.length).fill(0));
});

test('the shared cell crop feature extractor emits finite features', () => {
  const width = CELL_WINDOW_SIZE;
  const image = Buffer.alloc(width * width * 3);
  for (let y = 0; y < width; y++) {
    for (let x = 0; x < width; x++) {
      const index = (y * width + x) * 3;
      const value = (x % 18 < 2 || y % 42 < 2) ? 210 : 45 + ((x + y) % 24);
      image[index] = value;
      image[index + 1] = value;
      image[index + 2] = Math.min(255, value + 8);
    }
  }
  const features = cellCropFeatures(image, width, CELL_WINDOW_MARGIN);
  assert.equal(features.length, CELL_FEATURE_NAMES.length);
  assert.ok(features.every(Number.isFinite));
});
