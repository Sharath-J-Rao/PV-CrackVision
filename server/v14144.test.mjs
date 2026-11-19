import test from 'node:test';
import assert from 'node:assert/strict';
import { markedAnomalyStyle } from '../src/utils/markedImage.ts';
import { flipTensor } from './cellClassifier.ts';
import { normalizeTrainingSettings } from './trainingSettings.ts';

test('marked-image style: user cracks stay green, model cracks keep outer/low/ok colours (v1.4.14.4 type fix)', () => {
  assert.equal(markedAnomalyStyle({ category: 'spider_web_crack', source: 'user', confidence: 0.1 }).tag, 'USER');
  assert.equal(markedAnomalyStyle({ category: 'spider_web_crack', outerRing: true, confidence: 0.99 }).tag, 'OUTER');
  assert.equal(markedAnomalyStyle({ category: 'spider_web_crack', confidence: 0.5 }).tag, 'LOW');
  assert.equal(markedAnomalyStyle({ category: 'spider_web_crack', confidence: 0.95 }).tag, '');
});

test('flipTensor mirrors rows/columns per channel, is an involution, and identity returns the input', () => {
  const size = 3, src = Float32Array.from({ length: 27 }, (_, i) => i);
  assert.equal(flipTensor(src, size, { flipH: false, flipV: false }), src);
  const h = flipTensor(src, size, { flipH: true, flipV: false });
  assert.deepEqual([...h.slice(0, 3)], [2, 1, 0]);
  const v = flipTensor(src, size, { flipH: false, flipV: true });
  assert.deepEqual([...v.slice(0, 3)], [6, 7, 8]);
  assert.deepEqual([...v.slice(9, 12)], [15, 16, 17]); // channel 1 flips independently of channel 0
  const hv = flipTensor(src, size, { flipH: true, flipV: true });
  assert.deepEqual([...flipTensor(hv, size, { flipH: true, flipV: true })], [...src]);
});

test('candidateTta defaults on and old settings files pick the default up', () => {
  assert.equal(normalizeTrainingSettings({}).crackNet.candidateTta, true);
  assert.equal(normalizeTrainingSettings({ crackNet: { candidateTta: false } }).crackNet.candidateTta, false);
});
