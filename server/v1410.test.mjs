import test from 'node:test';
import assert from 'node:assert/strict';
import { trainCnn, cnnProbability, CNN_INPUT_SIZE } from './cellCnn.ts';
import { isPerimeterCell } from './cellPriority.ts';
import { detectCellBoxes, registerClassicalCellProvider, setCellDetectorMode } from './cellDetector.ts';
import { normalizeTrainingSettings } from './trainingSettings.ts';

const size = CNN_INPUT_SIZE;
function sample(index, y) {
  const input = new Float32Array(2 * size * size);
  for (let i = 0; i < input.length; i++) input[i] = Math.sin(index * 12.9898 + i * 78.233) * 0.3;
  if (y) for (let r = 18; r < 30; r++) for (let c = 18; c < 30; c++) { input[r * size + c] += 2; input[size * size + r * size + c] += 2; }
  return { input, y };
}

test('CNN restores the weights of the epoch with the highest held-out balanced accuracy', async () => {
  const train = Array.from({ length: 24 }, (_, i) => sample(i, i % 2));
  const validation = Array.from({ length: 12 }, (_, i) => sample(100 + i, i % 2));
  const result = await trainCnn(train, train.map(() => 1), { epochs: 5, batchSize: 8, earlyStopPatience: 0 }, undefined, validation);
  assert.equal(result.history.length, 5);
  const best = Math.max(...result.history.map((row) => row.balancedAccuracy));
  assert.equal(result.history[result.bestEpoch - 1].balancedAccuracy, best);
  let tp = 0, tn = 0, p = 0, n = 0;
  for (const s of validation) { const pred = cnnProbability(result.model, s.input) >= 0.5 ? 1 : 0; if (s.y) { p++; tp += pred; } else { n++; tn += 1 - pred; } }
  assert.equal(Number((((tp / p) + (tn / n)) / 2).toFixed(4)), best); // returned model == best epoch, not the last
});

test('CNN early stopping halts after the patience window and pause hook is awaited', async () => {
  const train = Array.from({ length: 16 }, (_, i) => sample(i, i % 2));
  const validation = Array.from({ length: 8 }, (_, i) => sample(200 + i, (i + 1) % 2 === 0 ? 1 : 0));
  let pauses = 0;
  const result = await trainCnn(train, train.map(() => 1), { epochs: 40, batchSize: 8, earlyStopPatience: 1, learningRate: 0.0001 }, undefined, validation, { waitIfPaused: async () => { pauses++; } });
  assert.ok(pauses > 0);
  assert.ok(result.history.length <= 40);
  if (result.history.length < 40) assert.equal(result.stoppedEarly, true);
});

test('perimeter cells are the outer ring of a module only', () => {
  const module = { left: 0, top: 0, width: 600, height: 400 };
  const cell = (left, top) => ({ left, top, width: 100, height: 100 });
  assert.equal(isPerimeterCell(cell(5, 150), module), true);
  assert.equal(isPerimeterCell(cell(495, 150), module), true);
  assert.equal(isPerimeterCell(cell(150, 5), module), true);
  assert.equal(isPerimeterCell(cell(200, 150), module), false);
  assert.equal(isPerimeterCell(cell(200, 150), undefined), false);
});

test('auto mode falls back to a real classical detector; explicit YOLO mode does not', async () => {
  registerClassicalCellProvider(async () => [{ left: 1, top: 1, width: 20, height: 20, confidence: 0.5 }, { left: 30, top: 1, width: 20, height: 20, confidence: 0.5 }]);
  const rgb = Buffer.alloc(64 * 64 * 3);
  setCellDetectorMode('auto');
  const auto = await detectCellBoxes(rgb, 64, 64);
  assert.equal(auto.source, 'classical'); assert.equal(auto.boxes.length, 2); assert.match(auto.warning, /classical/i);
  setCellDetectorMode('classical');
  assert.equal((await detectCellBoxes(rgb, 64, 64)).boxes.length, 2);
  setCellDetectorMode('yolo11n');
  const strict = await detectCellBoxes(rgb, 64, 64);
  assert.equal(strict.source, 'yolo11n'); assert.equal(strict.boxes.length, 0);
  setCellDetectorMode('auto');
});

test('new v1.4.10 settings are bounded and defaulted', () => {
  const s = normalizeTrainingSettings({ crackClassifier: { cnnEarlyStopPatience: 999, edgeThresholdDrop: -1, edgeSampleWeight: 99, hardNegativeWeight: 'x' } });
  assert.equal(s.crackClassifier.cnnEarlyStopPatience, 100);
  assert.equal(s.crackClassifier.edgeThresholdDrop, 0);
  assert.equal(s.crackClassifier.edgeSampleWeight, 10);
  assert.equal(s.crackClassifier.hardNegativeWeight, 2);
  assert.equal(normalizeTrainingSettings({}).cellDetector.patience, 30);
});
