// v1.4.12.6 regression tests: persistent review store, larger YOLO image sizes, and the training-status race.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// Use an isolated data root BEFORE any server module is imported (paths are resolved at import time).
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'pvcv-v14126-'));
process.env.PV_CRACKVISION_DATA_DIR = root;
const jobStateFile = path.join(root, 'training', '.cell-yolo-job-state.json');
fs.mkdirSync(path.dirname(jobStateFile), { recursive: true });
const baseState = { progress: 50, message: 'Training', log: [], jobId: 'job-1', pid: process.pid, updatedAt: new Date().toISOString() };
fs.writeFileSync(jobStateFile, JSON.stringify({ ...baseState, status: 'training' }));

const { loadReview, saveReview, deleteReview, isValidReviewHash } = await import('./reviewStore.ts');
const { normalizeTrainingSettings, DEFAULT_TRAINING_SETTINGS } = await import('./trainingSettings.ts');
const { getCellTrainingStatus } = await import('./cellTraining.ts');

const HASH = 'a'.repeat(40);
const cell = { status: 'completed', boxes: [{ left: 1, top: 2, width: 3, height: 4 }], reviewed: true };

test('review store round-trips cell and crack review and deletes it', () => {
  assert.equal(loadReview(HASH), null);
  saveReview(HASH, { fileName: 'C:\\flight\\img1.jpg', cell, crack: { result: { anomalies: [] }, rejected: [], crackReviewed: true } });
  const record = loadReview(HASH);
  assert.equal(record.fileName, 'img1.jpg', 'only the base name is stored');
  assert.deepEqual(record.cell.boxes, cell.boxes);
  assert.equal(record.crack.crackReviewed, true);
  saveReview(HASH, { fileName: 'img1.jpg', cell: { ...cell, boxes: [] } });
  assert.equal(loadReview(HASH).crack, undefined, 'a newer save replaces the whole record');
  assert.equal(deleteReview(HASH), true);
  assert.equal(deleteReview(HASH), false);
  assert.equal(loadReview(HASH), null);
});

test('review store rejects bad hashes (path traversal) and malformed payloads', () => {
  for (const bad of ['../../etc/passwd', 'A'.repeat(40), 'a'.repeat(39), '', undefined]) {
    assert.equal(isValidReviewHash(bad), false);
    assert.equal(loadReview(bad), null);
    assert.throws(() => saveReview(bad, { cell }), /Invalid image hash/);
  }
  assert.throws(() => saveReview(HASH, null), /must be an object/);
  assert.throws(() => saveReview(HASH, { fileName: 'x' }), /Nothing to save/);
  assert.throws(() => saveReview(HASH, { cell: { boxes: 'no' } }), /boxes array/);
  assert.throws(() => saveReview(HASH, { crack: { result: [] } }), /result object/);
  assert.equal(loadReview(HASH), null, 'nothing was written by the failed saves');
});

test('a corrupt saved review is ignored instead of crashing the load', () => {
  saveReview(HASH, { cell });
  fs.writeFileSync(path.join(root, 'training', 'review_state', `${HASH}.json`), '{not json');
  assert.equal(loadReview(HASH), null);
  deleteReview(HASH);
});

test('YOLO cell training accepts the larger sizes and still rejects unsupported ones', () => {
  for (const size of [960, 1280, 1536, 1920]) assert.equal(normalizeTrainingSettings({ cellDetector: { imageSize: size } }).cellDetector.imageSize, size);
  assert.equal(normalizeTrainingSettings({ cellDetector: { imageSize: 1000 } }).cellDetector.imageSize, DEFAULT_TRAINING_SETTINGS.cellDetector.imageSize);
  assert.equal(DEFAULT_TRAINING_SETTINGS.cellDetector.imageSize, 1280);
});

test('a finished record written by the worker for the running job is never replaced by stale in-memory state', () => {
  // Node still believes the job is training (live pid) ...
  assert.equal(getCellTrainingStatus().status, 'training');
  // ... the worker then writes its final record with an OLDER timestamp than Node's copy (clock/ordering skew).
  fs.writeFileSync(jobStateFile, JSON.stringify({ ...baseState, status: 'completed', progress: 100, updatedAt: '2000-01-01T00:00:00.000Z' }));
  const status = getCellTrainingStatus();
  assert.equal(status.status, 'completed');
  assert.equal(status.error, undefined, 'no false "worker is no longer running" failure');
});
