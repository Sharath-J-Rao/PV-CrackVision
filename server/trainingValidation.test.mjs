import test from 'node:test';
import assert from 'node:assert/strict';
import { getFeedbackReadiness, selectLabelCandidates, shouldAcceptCandidate, splitByImage, TRAINING_REQUIREMENTS } from './trainingValidation.ts';

test('training readiness checks class balance, image diversity and annotated PV regions', () => {
  const items = [];
  for (let image = 0; image < 10; image++) {
    for (let n = 0; n < (image < 3 ? 10 : 3); n++) {
      items.push({ imageFile: `image-${image}`, label: 1, moduleLeft: 0, moduleTop: 0, moduleWidth: 100, moduleHeight: 100 });
      items.push({ imageFile: `image-${image}`, label: 0, moduleLeft: 120, moduleTop: 0, moduleWidth: 100, moduleHeight: 100 });
    }
  }
  const status = getFeedbackReadiness(items);
  assert.equal(status.ready, true);
  assert.equal(status.images, 10);
  assert.equal(status.positiveImages, 10);
  assert.equal(status.negativeImages, 10);
  assert.equal(status.annotatedRegions, 20);
  assert.equal(TRAINING_REQUIREMENTS.positives, 30);
});

test('training readiness rejects unbalanced or single-image feedback', () => {
  const status = getFeedbackReadiness(Array.from({ length: 40 }, (_, i) => ({
    imageFile: 'one-image', label: i < 35 ? 1 : 0,
  })));
  assert.equal(status.ready, false);
  assert.ok(status.missing.some((entry) => entry.includes('distinct images')));
  assert.ok(status.missing.some((entry) => entry.includes('annotated PV regions')));
});

test('negative hard-example mining selects highest-scoring false positives', () => {
  const selected = selectLabelCandidates([{ score: 0.1 }, { score: 0.95 }, { score: 0.7 }], 2);
  assert.deepEqual(selected.map((item) => item.score), [0.95, 0.7]);
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


test('held-out gate rejects a candidate that harms either class', () => {
  const before = { recall: 0.8, specificity: 0.9, balancedAccuracy: 0.85, logLoss: 0.4 };
  const regressed = { recall: 0.7, specificity: 0.95, balancedAccuracy: 0.825, logLoss: 0.2 };
  assert.equal(shouldAcceptCandidate(before, regressed), false);
});

test('held-out gate accepts a real gain with no material class regression', () => {
  const before = { recall: 0.8, specificity: 0.9, balancedAccuracy: 0.85, logLoss: 0.4 };
  const improved = { recall: 0.82, specificity: 0.92, balancedAccuracy: 0.87, logLoss: 0.35 };
  assert.equal(shouldAcceptCandidate(before, improved), true);
});
