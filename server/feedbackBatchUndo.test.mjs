import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';

const dataRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'pv-feedback-undo-'));
process.env.PV_CRACKVISION_DATA_DIR = dataRoot;
const store = await import('./feedbackStore.ts');

const image = Buffer.from('unit-test-image-bytes');
const meta = (left, label, extra = {}) => ({ fileName: 'source.jpg', left, top: 10, width: 20, height: 20, label, category: 'spider_web_crack', ...extra });

test('v1.4.16.0 bulk-negative transactional update restores the previous cell label and score', () => {
  store.initializeCellFeedbackStore();
  const original = store.addFeedback(image, meta(10, 1, { source: 'missed-crack' }));
  const batch = store.addFeedbackManyTransactional(image, [meta(10, 0, { source: 'bulk-negative', modelScore: 0.0049 })]);
  assert.equal(batch.saved.length, 1);
  assert.equal(batch.saved[0].id, original.id, 'same cell is deduplicated rather than duplicated');
  assert.equal(store.listFeedback().find((item) => item.id === original.id).label, 0);
  assert.equal(store.listFeedback().find((item) => item.id === original.id).modelScore, 0.0049);
  assert.equal(store.undoFeedbackBatch(batch.undoToken), true);
  const restored = store.listFeedback().find((item) => item.id === original.id);
  assert.equal(restored.label, 1);
  assert.equal(restored.source, 'missed-crack');
  assert.equal(restored.modelScore, undefined);
  assert.equal(store.undoFeedbackBatch(batch.undoToken), false, 'undo token is one-shot');
});

test('v1.4.16.0 undo restores replaced rows and removes all rows created by the batch', () => {
  store.clearFeedbackDataset();
  store.addFeedbackMany(image, [meta(20, 1), meta(80, 0)]);
  const before = store.listFeedback().map((item) => item.id).sort();
  const batch = store.addFeedbackManyTransactional(image, [meta(150, 0, { source: 'bulk-negative' })], { replace: true });
  assert.equal(store.listFeedback().length, 1);
  assert.equal(store.undoFeedbackBatch(batch.undoToken), true);
  assert.deepEqual(store.listFeedback().map((item) => item.id).sort(), before);
});


test('v1.4.16.1 oversized replace batches are rejected before changing the feedback index', () => {
  store.clearFeedbackDataset();
  const imageFile = `${crypto.createHash('sha1').update(image).digest('hex').slice(0, 20)}.img`;
  const rows = Array.from({ length: 10001 }, (_, index) => ({
    id: `00000000-0000-4000-8000-${String(index + 1).padStart(12, '0')}`,
    fileName: 'source.jpg', imageFile, left: index * 25, top: 10, width: 20, height: 20,
    label: 1, category: 'spider_web_crack', createdAt: new Date().toISOString(),
  }));
  const indexPath = path.join(store.FEEDBACK_DIR, 'feedback.json');
  fs.writeFileSync(indexPath, JSON.stringify(rows));
  const before = fs.readFileSync(indexPath, 'utf8');
  assert.throws(() => store.addFeedbackManyTransactional(image, [meta(1, 0)], { replace: true }), /more than 10,000/);
  assert.equal(fs.readFileSync(indexPath, 'utf8'), before, 'existing annotations remain byte-for-byte unchanged');
  assert.equal(fs.existsSync(path.join(store.FEEDBACK_DIR, '.feedback-undo.json')), false, 'no unusable undo journal is written');
});

test.after(() => fs.rmSync(dataRoot, { recursive: true, force: true }));

test('v1.4.16.1 undo journal survives a module/runtime restart', async () => {
  store.clearFeedbackDataset();
  store.addFeedback(image, meta(230, 1, { source: 'missed-crack' }));
  const batch = store.addFeedbackManyTransactional(image, [meta(230, 0, { source: 'bulk-negative', modelScore: 0.91 })]);
  const restartedStore = await import(`./feedbackStore.ts?restart=${Date.now()}`);
  assert.equal(restartedStore.getLastFeedbackUndo()?.token, batch.undoToken, 'the app can recover the last undo token after restart');
  assert.equal(restartedStore.undoFeedbackBatch(batch.undoToken), true);
  const restored = restartedStore.listFeedback().find((item) => item.left === 230);
  assert.equal(restored.label, 1);
  assert.equal(restored.source, 'missed-crack');
});

test('v1.4.16.1 stale undo refuses to overwrite a newer edit to the same cell', () => {
  store.clearFeedbackDataset();
  const original = store.addFeedback(image, meta(280, 1, { source: 'missed-crack' }));
  const batch = store.addFeedbackManyTransactional(image, [meta(280, 0, { source: 'bulk-negative' })]);
  store.addFeedback(image, meta(280, 1, { source: 'missed-crack', modelScore: 0.25 }));
  assert.equal(store.undoFeedbackBatch(batch.undoToken), false);
  assert.equal(store.getLastFeedbackUndo(), null, 'the stale token is not offered again');
  const latest = store.listFeedback().find((item) => item.id === original.id);
  assert.equal(latest.label, 1);
  assert.equal(latest.modelScore, 0.25);
});

test('v1.4.16.1 refuses corrupt feedback JSON rather than treating it as an empty dataset', () => {
  store.clearFeedbackDataset();
  store.addFeedback(image, meta(330, 0));
  const indexPath = path.join(store.FEEDBACK_DIR, 'feedback.json');
  fs.writeFileSync(indexPath, '{not valid json');
  assert.throws(() => store.listFeedback(), /corrupt or unreadable/);
  assert.equal(fs.readFileSync(indexPath, 'utf8'), '{not valid json');
});

test('v1.4.16.1 feedback image path rejects path traversal', () => {
  assert.throws(() => store.feedbackImagePath({ imageFile: '../../outside.img' }), /Invalid feedback image identifier/);
});

test('v1.4.16.1 corrupt undo journal fails closed instead of hiding or replacing invalid entries', () => {
  store.clearFeedbackDataset();
  store.addFeedback(image, meta(390, 1, { source: 'missed-crack' }));
  const batch = store.addFeedbackManyTransactional(image, [meta(390, 0, { source: 'bulk-negative' })]);
  const journalPath = path.join(store.FEEDBACK_DIR, '.feedback-undo.json');
  fs.writeFileSync(journalPath, JSON.stringify([{ token: batch.undoToken, createdAt: Date.now(), changes: [{ id: 'not-a-uuid', before: null, after: null }] }]));
  assert.throws(() => store.getLastFeedbackUndo(), /invalid records/);
  assert.equal(store.listFeedback().find((item) => item.left === 390).label, 0);
});

test('v1.4.16.1 undo order exposes the previous batch again after undoing an overlapping newer batch', () => {
  store.clearFeedbackDataset();
  store.addFeedback(image, meta(450, 1, { source: 'missed-crack' }));
  const first = store.addFeedbackManyTransactional(image, [meta(450, 0, { source: 'bulk-negative' })]);
  const second = store.addFeedbackManyTransactional(image, [meta(450, 1, { source: 'missed-crack' })]);
  assert.equal(store.getLastFeedbackUndo()?.token, second.undoToken);
  assert.equal(store.undoFeedbackBatch(second.undoToken), true);
  assert.equal(store.listFeedback().find((item) => item.left === 450).label, 0);
  assert.equal(store.getLastFeedbackUndo()?.token, first.undoToken);
  assert.equal(store.undoFeedbackBatch(first.undoToken), true);
  assert.equal(store.listFeedback().find((item) => item.left === 450).label, 1);
});
