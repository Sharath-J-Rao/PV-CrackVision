import test from 'node:test';
import assert from 'node:assert/strict';
import { getFeedbackReadiness, splitByImage, TRAINING_REQUIREMENTS } from './trainingValidation.ts';

test('training readiness checks class balance, image diversity and annotated PV regions', () => {
  const items = [];
  for (let image = 0; image < 4; image++) {
    for (let n = 0; n < 4; n++) {
      items.push({ imageFile: `image-${image}`, label: 1, moduleLeft: 0, moduleTop: 0, moduleWidth: 100, moduleHeight: 100 });
      items.push({ imageFile: `image-${image}`, label: 0, moduleLeft: 120, moduleTop: 0, moduleWidth: 100, moduleHeight: 100 });
    }
  }
  const status = getFeedbackReadiness(items);
  assert.equal(status.ready, true);
  assert.equal(status.images, 4);
  assert.equal(status.positiveImages, 4);
  assert.equal(status.negativeImages, 4);
  assert.equal(status.annotatedRegions, 8);
  assert.equal(TRAINING_REQUIREMENTS.positives, 8);
});

test('training readiness rejects unbalanced or single-image feedback', () => {
  const status = getFeedbackReadiness(Array.from({ length: 40 }, (_, i) => ({
    imageFile: 'one-image', label: i < 35 ? 1 : 0,
  })));
  assert.equal(status.ready, false);
  assert.ok(status.missing.some((entry) => entry.includes('distinct images')));
  assert.ok(status.missing.some((entry) => entry.includes('annotated PV regions')));
});

test('image-level split never leaks samples from the same image', () => {
  const groups = Array.from({ length: 12 }, (_, image) => ({
    imageFile: `image-${image}`,
    samples: [
      { y: 1, imageFile: `image-${image}` },
      { y: 0, imageFile: `image-${image}` },
    ],
  }));
  const split = splitByImage(groups);
  assert.ok(split.validationImages.length >= 2);
  assert.ok(split.trainImages.length >= 2);
  assert.ok(split.train.every((sample) => !split.validationImages.includes(sample.imageFile)));
  assert.ok(split.validation.every((sample) => !split.trainImages.includes(sample.imageFile)));
  assert.ok(split.train.some((sample) => sample.y === 1) && split.train.some((sample) => sample.y === 0));
  assert.ok(split.validation.some((sample) => sample.y === 1) && split.validation.some((sample) => sample.y === 0));
});
