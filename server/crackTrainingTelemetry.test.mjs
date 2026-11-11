import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { trainCnn, CNN_INPUT_SIZE } from './cellCnn.ts';
import { trainGbm, trainGbmAsync } from './gradientBoosting.ts';

test('CNN records train loss and held-out validation metrics on every epoch', async () => {
  const S = CNN_INPUT_SIZE;
  const create = (label, index) => {
    const input = new Float32Array(2 * S * S);
    for (let i = 0; i < S * S; i++) {
      input[i] = ((i * 13 + index * 7) % 29) / 29 - 0.5;
      input[S * S + i] = label ? ((i % S) === Math.floor(i / S) ? 0.9 : 0.05) : 0.02;
    }
    return { input, y: label ? 1 : 0 };
  };
  const train = Array.from({ length: 20 }, (_, i) => create(i % 2 === 0, i));
  const validation = Array.from({ length: 8 }, (_, i) => create(i % 2 === 0, i + 100));
  const seen = [];
  const result = await trainCnn(train, train.map(() => 1), { epochs: 3, batchSize: 8 }, (epoch, loss, metrics) => {
    seen.push({ epoch, loss, metrics });
  }, validation);
  assert.equal(result.history.length, 3);
  assert.equal(seen.length, 3);
  for (const row of result.history) {
    assert.ok(Number.isFinite(row.loss));
    assert.ok(Number.isFinite(row.validationLoss));
    assert.ok(Number.isFinite(row.precision));
    assert.ok(Number.isFinite(row.recall));
    assert.ok(Number.isFinite(row.specificity));
    assert.ok(Number.isFinite(row.balancedAccuracy));
    assert.ok(row.balancedAccuracy >= 0 && row.balancedAccuracy <= 1);
    assert.ok(Number.isFinite(row.f1));
    assert.ok(row.f1 >= 0 && row.f1 <= 1);
  }
});

test('crack-classifier job state persists stage, epoch metrics, and log/CSV artifacts', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pvcv-crack-job-'));
  const previous = process.env.PV_CRACKVISION_DATA_DIR;
  process.env.PV_CRACKVISION_DATA_DIR = dir;
  try {
    const jobs = await import(`./crackClassifierTrainingJob.ts?test=${Date.now()}`);
    jobs.beginCrackTrainingJob('job-test', 'ensemble', 3, 'Preparing test job');
    jobs.updateCrackTrainingJob({ status: 'training', stage: 'cnn', progress: 66, epoch: 2, epochs: 3, message: 'CNN epoch 2 complete' });
    jobs.addCrackEpochMetric({ epoch: 2, trainLoss: 0.32, validationLoss: 0.41, precision: 0.8, recall: 0.75, specificity: 0.9, balancedAccuracy: 0.825, f1: 0.7742, component: 'CNN' });
    jobs.appendCrackTrainingLog('test epoch persisted');
    const state = jobs.getCrackTrainingJobState();
    assert.equal(state.status, 'training');
    assert.equal(state.epoch, 2);
    assert.equal(state.metrics.length, 1);
    assert.match(jobs.crackTrainingMetricsCsv(), /validation_loss/);
    assert.match(jobs.crackTrainingMetricsCsv(), /f1/);
    assert.match(jobs.crackTrainingMetricsCsv(), /0\.7742/);
    assert.match(jobs.crackTrainingMetricsCsv(), /0\.41/);
    assert.match(jobs.crackTrainingLogText(), /test epoch persisted/);
    assert.ok(fs.existsSync(path.join(dir, 'training', 'crack_classifier', 'job-state.json')));
    assert.ok(fs.existsSync(path.join(dir, 'training', 'crack_classifier', 'training.log')));
    assert.ok(fs.existsSync(path.join(dir, 'training', 'crack_classifier', 'epoch-metrics.csv')));
  } finally {
    if (previous === undefined) delete process.env.PV_CRACKVISION_DATA_DIR;
    else process.env.PV_CRACKVISION_DATA_DIR = previous;
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('GBM training yields after every tree so live training status can be polled', async () => {
  const X = Array.from({ length: 24 }, (_, i) => [i / 24, (i % 5) / 5, i % 2]);
  const y = Array.from({ length: 24 }, (_, i) => (i % 2 ? 1 : 0));
  const seen = [];
  const params = { trees: 4, depth: 2, minLeaf: 2 };
  const model = await trainGbmAsync(X, y, y.map(() => 1), params, (tree, total) => seen.push([tree, total]));
  assert.deepEqual(model, trainGbm(X, y, y.map(() => 1), params), 'async telemetry path must preserve the deterministic GBM model');
  assert.equal(model.trees.length, 4);
  assert.deepEqual(seen, [[1, 4], [2, 4], [3, 4], [4, 4]]);
});
