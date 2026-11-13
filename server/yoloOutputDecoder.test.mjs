import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeYoloCellOutput } from './yoloOutputDecoder.ts';

function tensor5xN(candidates) {
  const data = new Float32Array(5 * candidates.length);
  for (let i = 0; i < candidates.length; i++) {
    for (let channel = 0; channel < 5; channel++) data[channel * candidates.length + i] = candidates[i][channel];
  }
  return { dims: [1, 5, candidates.length], data };
}

test('decodes the logged Ultralytics single-class output shape [1,5,18900]', () => {
  const values = Array.from({ length: 18900 }, () => [0, 0, 0, 0, 0.001]);
  values[123] = [480, 360, 80, 60, 0.17];
  values[456] = [620, 360, 80, 60, 0.61];
  const result = decodeYoloCellOutput(tensor5xN(values), 960, 720, 1, 0, 0);
  assert.equal(result.diagnostics.outputShape, '1x5x18900');
  assert.equal(result.diagnostics.candidatesAboveThreshold, 2);
  assert.equal(result.boxes.length, 2);
  assert.ok(Math.abs(result.boxes[0].confidence - 0.61) < 1e-6);
  assert.equal(result.boxes[0].left, 580);
  assert.equal(result.boxes[0].top, 330);
});

test('uses the same decoder for transposed [1,N,5] output', () => {
  const result = decodeYoloCellOutput({
    dims: [1, 2, 5],
    data: new Float32Array([100, 100, 40, 40, 0.7, 200, 100, 40, 40, 0.08]),
  }, 300, 200, 1, 0, 0);
  assert.equal(result.boxes.length, 2);
});

test('decodes a one-candidate channel-first tensor [1,5,1]', () => {
  const result = decodeYoloCellOutput(tensor5xN([[100, 100, 40, 40, 0.02]]), 300, 200, 1, 0, 0, 0.01);
  assert.equal(result.boxes.length, 1);
});

test('reports a truly low-confidence output rather than silently inventing boxes', () => {
  const result = decodeYoloCellOutput(tensor5xN([[100, 100, 40, 40, 0.02]]), 300, 200, 1, 0, 0, 0.05);
  assert.equal(result.boxes.length, 0);
  assert.equal(result.diagnostics.candidatesAboveThreshold, 0);
  assert.ok(Math.abs(result.diagnostics.maximumScore - 0.02) < 1e-6);
});
