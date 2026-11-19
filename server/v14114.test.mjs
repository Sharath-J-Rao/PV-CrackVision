import test from 'node:test';
import assert from 'node:assert/strict';
import { computeClassWeights } from './trainingValidation.ts';

const mk = (y, mult = 1) => ({ y, mult });
const mass = (samples, weights, y) => samples.reduce((s, x, i) => s + (x.y === y ? weights[i] : 0), 0);

test('cracked and clean cells carry equal total weight on a 1:9 imbalance', () => {
  const samples = [...Array(10).fill(0).map(() => mk(1)), ...Array(90).fill(0).map(() => mk(0))];
  const { weights, summary } = computeClassWeights(samples, true);
  assert.ok(Math.abs(mass(samples, weights, 1) - mass(samples, weights, 0)) < 1e-9);
  assert.equal(summary.imbalanceRatio, 9);
  assert.ok(summary.positiveWeight > summary.negativeWeight);
  assert.ok(Math.abs(weights.reduce((a, b) => a + b, 0) - 100) < 1e-9, 'total weight is preserved');
});

test('edge / hard-negative multipliers no longer re-skew the class balance', () => {
  const samples = [mk(1, 3), mk(1, 3), mk(0), mk(0), mk(0, 4), mk(0, 4)]; // positives are edge cells, some negatives are hard negatives
  const { weights } = computeClassWeights(samples, true);
  assert.ok(Math.abs(mass(samples, weights, 1) - mass(samples, weights, 0)) < 1e-9);
  assert.ok(weights[4] > weights[2], 'a hard negative still outweighs an ordinary negative of the same class');
});

test('disabled balancing leaves only the multipliers; a missing class cannot be balanced', () => {
  const samples = [mk(1, 2), mk(0)];
  assert.deepEqual(computeClassWeights(samples, false).weights, [2, 1]);
  const none = computeClassWeights([mk(0), mk(0)], true);
  assert.deepEqual(none.weights, [1, 1]);
  assert.equal(none.summary.imbalanceRatio, null);
});
