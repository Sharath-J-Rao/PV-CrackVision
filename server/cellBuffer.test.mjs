import test from 'node:test';
import assert from 'node:assert/strict';
import { bufferedCellCrop, extractCellFeatures, extractCellSample, DEFAULT_CELL_BUFFER } from './crackModel.ts';
import { normalizeTrainingSettings, DEFAULT_TRAINING_SETTINGS } from './trainingSettings.ts';

const box = { left: 100, top: 100, width: 100, height: 100 };

test('buffer 0 reproduces the exact cell box (old models score exactly as before)', () => {
  assert.deepEqual(bufferedCellCrop(1000, 1000, box, 0), { left: 100, top: 100, width: 100, height: 100 });
  assert.deepEqual(bufferedCellCrop(1000, 1000, box), { left: 100, top: 100, width: 100, height: 100 });
});

test('default buffer is minimal: 3% of the short side, at least 1 px, grown on every side', () => {
  assert.equal(DEFAULT_CELL_BUFFER, 0.03);
  assert.deepEqual(bufferedCellCrop(1000, 1000, box, 0.03), { left: 97, top: 97, width: 106, height: 106 });
  const small = bufferedCellCrop(1000, 1000, { left: 50, top: 50, width: 20, height: 20 }, 0.03); // 0.6 px -> 1 px minimum
  assert.deepEqual(small, { left: 49, top: 49, width: 22, height: 22 });
});

test('buffer is clamped at image borders', () => {
  assert.deepEqual(bufferedCellCrop(205, 200, { left: 2, top: 0, width: 200, height: 200 }, 0.03), { left: 0, top: 0, width: 205, height: 200 });
});

test('settings: cellBuffer defaults to 0.03 and is clamped to 0..0.1', () => {
  assert.equal(DEFAULT_TRAINING_SETTINGS.crackClassifier.cellBuffer, 0.03);
  assert.equal(normalizeTrainingSettings({ crackClassifier: { cellBuffer: 5 } }).crackClassifier.cellBuffer, 0.1);
  assert.equal(normalizeTrainingSettings({ crackClassifier: { cellBuffer: -1 } }).crackClassifier.cellBuffer, 0);
  assert.equal(normalizeTrainingSettings({ crackClassifier: { cellBuffer: 0 } }).crackClassifier.cellBuffer, 0);
});

test('check: a slightly tight detection clips an edge crack; the buffer recovers it, and buffer 0 is byte-identical to the legacy path', async () => {
  const W = 400, H = 400;
  const rgb = Buffer.alloc(W * H * 3, 120); // flat grey cell surface
  // dark crack line running along the true cell's left edge, x = 118..119 (true cell = left 118..), detection starts 2 px too far right
  for (let y = 110; y < 190; y++) for (let x = 117; x < 120; x++) { const i = (y * W + x) * 3; rgb[i] = rgb[i + 1] = rgb[i + 2] = 20; }
  const tight = { left: 120, top: 110, width: 80, height: 80 };
  const legacy = await extractCellFeatures(rgb, W, H, tight);
  const noBuffer = await extractCellSample(rgb, W, H, tight, { buffer: 0 });
  assert.deepEqual(noBuffer.features, legacy);
  const buffered = await extractCellSample(rgb, W, H, tight, { buffer: 0.03 }); // 2 px
  assert.notDeepEqual(buffered.features, legacy, 'the buffered crop sees the boundary crack the tight crop missed');
  const flat = legacy.every((v) => Number.isFinite(v));
  assert.ok(flat && buffered.features.every((v) => Number.isFinite(v)));
});
