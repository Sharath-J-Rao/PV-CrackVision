import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { saveCellAnnotations, saveNegativeCellSample, listCellAnnotations } from './cellAnnotationStore.ts';
import { CELL_ANNOTATIONS_DIR } from './dataPaths.ts';

const root = CELL_ANNOTATIONS_DIR;
const imageDir = path.join(root, 'images');
const indexPath = path.join(root, 'annotations.json');

test('re-saving an image replaces the full reviewed box set so deleted detections stay deleted', () => {
  const image = Buffer.from(`annotation-store-regression-${process.pid}-${Date.now()}`);
  const first = saveCellAnnotations(image, 'same-source.jpg', [
    { left: 10, top: 10, width: 40, height: 50 },
    { left: 70, top: 10, width: 40, height: 50 },
  ]);
  const second = saveCellAnnotations(image, 'same-source.jpg', [
    { left: 10, top: 10, width: 40, height: 50 }, // same cell: update, do not duplicate
    { left: 130, top: 10, width: 40, height: 50 }, // newly annotated cell
  ]);
  assert.equal(second.id, first.id);
  assert.equal(second.boxes.length, 2);
  assert.deepEqual(second.boxes.map((box) => box.left).sort((a, b) => a - b), [10, 130]);

  // Isolate the test from any operator data in a developer's working tree.
  const records = listCellAnnotations().filter((record) => record.imageFile !== first.imageFile);
  fs.writeFileSync(indexPath, JSON.stringify(records, null, 2));
  fs.rmSync(path.join(imageDir, first.imageFile), { force: true });
});


test('background-only negatives persist as empty-label detector samples and can later be positively annotated', () => {
  const image = Buffer.from(`negative-sample-regression-${process.pid}-${Date.now()}`);
  const negative = saveNegativeCellSample(image, 'roof-background.jpg');
  assert.equal(negative.sampleType, 'negative');
  assert.equal(negative.boxes.length, 0);

  const positive = saveCellAnnotations(image, 'roof-background.jpg', [
    { left: 12, top: 18, width: 36, height: 42 },
  ]);
  assert.equal(positive.id, negative.id);
  assert.equal(positive.sampleType, 'positive');
  assert.equal(positive.boxes.length, 1);

  const records = listCellAnnotations().filter((record) => record.imageFile !== negative.imageFile);
  fs.writeFileSync(indexPath, JSON.stringify(records, null, 2));
  fs.rmSync(path.join(imageDir, negative.imageFile), { force: true });
});


test('refuses to reclassify an image with saved positive cell boxes as a whole-image negative', () => {
  const image = Buffer.from(`positive-negative-conflict-${process.pid}-${Date.now()}`);
  const positive = saveCellAnnotations(image, 'annotated-pv.jpg', [
    { left: 12, top: 18, width: 36, height: 42 },
  ]);
  assert.throws(() => saveNegativeCellSample(image, 'annotated-pv.jpg'), /already has positive cell boxes/);
  const records = listCellAnnotations().filter((record) => record.imageFile !== positive.imageFile);
  fs.writeFileSync(indexPath, JSON.stringify(records, null, 2));
  fs.rmSync(path.join(imageDir, positive.imageFile), { force: true });
});
