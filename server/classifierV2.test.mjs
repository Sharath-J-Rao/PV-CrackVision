import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { cellCropFeatures, CELL_FEATURE_NAMES, CELL_FEATURE_NAMES_V2, CELL_WINDOW_SIZE, CELL_WINDOW_MARGIN } from './cellFeatures.ts';
import { trainGbm, gbmProbability, isValidGbm } from './gradientBoosting.ts';
import { trainCnn, cnnProbability, buildCnnInput, dihedral, isValidCnn, CNN_INPUT_SIZE } from './cellCnn.ts';

function cellImage(crack) {
  const D = CELL_WINDOW_SIZE;
  const img = Buffer.alloc(D * D * 3);
  let seed = crack ? 5 : 9;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  for (let y = 0; y < D; y++) for (let x = 0; x < D; x++) {
    let v = 120 + rnd() * 6;
    if (crack) {
      const onLine = Math.abs((x - 40) - (y - 40) * 0.9) < 1.6 && y > 40 && y < 170;
      if (onLine) v = 165; // bright hairline (ridge response is a top-hat on bright thin structure), below the 175 gridline cut
    }
    const i = (y * D + x) * 3;
    img[i] = v; img[i + 1] = v; img[i + 2] = v + 4;
  }
  return img;
}

test('rich features: 16 base features unchanged, 22 extra finite features, crack raises continuity/diagonal evidence', () => {
  const clean = cellImage(false), cracked = cellImage(true);
  const base = cellCropFeatures(cracked, CELL_WINDOW_SIZE, CELL_WINDOW_MARGIN);
  const rich = cellCropFeatures(cracked, CELL_WINDOW_SIZE, CELL_WINDOW_MARGIN, true);
  assert.equal(base.length, CELL_FEATURE_NAMES.length);
  assert.equal(rich.length, CELL_FEATURE_NAMES_V2.length);
  assert.deepEqual(rich.slice(0, base.length), base);
  assert.ok(rich.every(Number.isFinite));
  const richClean = cellCropFeatures(clean, CELL_WINDOW_SIZE, CELL_WINDOW_MARGIN, true);
  const idx = (n) => CELL_FEATURE_NAMES_V2.indexOf(n);
  assert.ok(rich[idx('cont_span')] > richClean[idx('cont_span')]);
  assert.ok(rich[idx('max_run_frac')] > richClean[idx('max_run_frac')]);
  assert.ok(rich[idx('ori_diag')] > richClean[idx('ori_diag')] + 0.2);
  assert.ok(rich[idx('line_frac')] > richClean[idx('line_frac')]);
  const hist = rich.slice(idx('ori_0'), idx('ori_7') + 1);
  assert.ok(Math.abs(hist.reduce((a, b) => a + b, 0) - 1) < 1e-6);
});

test('gradient boosting learns a non-linear boundary and survives a JSON round-trip', () => {
  let seed = 3;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  const make = (n) => { const X = [], y = []; for (let i = 0; i < n; i++) { const a = rnd(), b = rnd(); X.push([a, b, rnd()]); y.push((a > 0.5) !== (b > 0.5) ? 1 : 0); } return { X, y }; };
  const train = make(400), test_ = make(200);
  const model = trainGbm(train.X, train.y, train.y.map(() => 1), { trees: 80, depth: 3 });
  const copy = JSON.parse(JSON.stringify(model));
  assert.ok(isValidGbm(copy, 3));
  let correct = 0;
  test_.X.forEach((x, i) => { if ((gbmProbability(copy, x) >= 0.5 ? 1 : 0) === test_.y[i]) correct++; });
  assert.ok(correct / test_.y.length > 0.9, `accuracy ${correct / test_.y.length}`);
  assert.equal(isValidGbm({ ...copy, featureCount: 5 }, 3), false);
});

test('small CNN trains on cell-crop tensors, generalises, and its dihedral augmentation is a bijection', async () => {
  const S = CNN_INPUT_SIZE;
  let seed = 21;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  const sample = (crack) => {
    const input = new Float32Array(2 * S * S);
    for (let i = 0; i < S * S; i++) { input[i] = (rnd() - 0.5) * 0.4; input[S * S + i] = rnd() * 0.1; }
    if (crack) {
      const o = Math.floor(rnd() * 20) + 10;
      for (let k = 0; k < 28; k++) { const x = o + k, y = o + k; input[y * S + x] = -2; input[S * S + y * S + x] = 2.5; }
    }
    return { input, y: crack ? 1 : 0 };
  };
  const trainSet = Array.from({ length: 80 }, (_, i) => sample(i % 2 === 0));
  const testSet = Array.from({ length: 40 }, (_, i) => sample(i % 2 === 0));
  const { model, history } = await trainCnn(trainSet, trainSet.map(() => 1), { epochs: 8, batchSize: 16 });
  assert.ok(isValidCnn(JSON.parse(JSON.stringify(model))));
  assert.ok(history[history.length - 1].loss < history[0].loss, 'loss must fall');
  let correct = 0;
  for (const t of testSet) if ((cnnProbability(model, t.input) >= 0.5 ? 1 : 0) === t.y) correct++;
  assert.ok(correct / testSet.length >= 0.85, `accuracy ${correct / testSet.length}`);
  const probe = trainSet[0].input;
  for (let k = 0; k < 8; k++) {
    const t = dihedral(probe, S, k);
    assert.equal([...t].sort().join(), [...probe].sort().join());
  }
});

test('buildCnnInput is finite, bounded and deterministic for a real extractor capture', () => {
  const capture = {};
  cellCropFeatures(cellImage(true), CELL_WINDOW_SIZE, CELL_WINDOW_MARGIN, true, capture);
  const a = buildCnnInput(capture), b = buildCnnInput(capture);
  assert.equal(a.length, 2 * CNN_INPUT_SIZE * CNN_INPUT_SIZE);
  assert.ok(a.every((v) => Number.isFinite(v) && v >= -3 && v <= 3));
  assert.deepEqual(a, b);
  assert.ok(Math.max(...a.slice(CNN_INPUT_SIZE * CNN_INPUT_SIZE)) > 0.5, 'the crack must survive down-sampling in the ridge channel');
});

test('a v2 model file round-trips through readModel/scoreCell; broken files are rejected', async () => {
  const { readModel, scoreCell } = await import('./crackModel.ts');
  const X = Array.from({ length: 60 }, (_, i) => Array.from({ length: CELL_FEATURE_NAMES_V2.length }, (_, j) => (i % 2) * (j === 3 ? 2 : 0) + ((i * 7 + j) % 5) * 0.1));
  const y = X.map((_, i) => i % 2);
  const gbm = trainGbm(X, y, y.map(() => 1), { trees: 20 });
  const model = { schema: 'cell-level-v2', version: 't', window: CELL_WINDOW_SIZE, stride: CELL_WINDOW_SIZE, margin: CELL_WINDOW_MARGIN, featureNames: [...CELL_FEATURE_NAMES_V2], mean: [], scale: [], coef: [], intercept: 0, threshold: 0.5, algorithm: 'gbm', gbm };
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pvcv-'));
  const file = path.join(dir, 'm.json');
  fs.writeFileSync(file, JSON.stringify(model));
  const loaded = readModel(file);
  assert.ok(loaded);
  assert.ok(scoreCell(loaded, X[1]) > 0.5 && scoreCell(loaded, X[0]) < 0.5);
  fs.writeFileSync(file, JSON.stringify({ ...model, gbm: { ...gbm, featureCount: 3 } }));
  assert.equal(readModel(file), null);
  fs.writeFileSync(file, JSON.stringify({ ...model, gbm: undefined }));
  assert.equal(readModel(file), null);
  fs.rmSync(dir, { recursive: true, force: true });
});
