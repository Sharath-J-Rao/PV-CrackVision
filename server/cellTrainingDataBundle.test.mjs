import test from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { createCellTrainingDataBundle, parseCellTrainingDataBundle } from './cellTrainingDataBundle.ts';

const sourceImage = Buffer.from('PV CrackVision test image bytes');
const imageFile = `${crypto.createHash('sha1').update(sourceImage).digest('hex').slice(0, 20)}.img`;
const record = {
  id: 'fixture-record', fileName: 'legacy-image.jpg', imageFile,
  boxes: [{ left: 1, top: 2, width: 30, height: 40 }],
  missedBoxes: [{ left: 5, top: 6, width: 10, height: 12 }],
  sampleType: 'positive', partial: true, createdAt: '2026-01-01T00:00:00.000Z',
};

test('cell training backup round-trips annotations and verified source-image bytes', async () => {
  const archive = await createCellTrainingDataBundle([record], () => sourceImage, '1.4.15-0');
  const restored = await parseCellTrainingDataBundle(archive);
  assert.equal(restored.manifest.format, 'pv-crackvision-cell-training-data');
  assert.equal(restored.manifest.appVersion, '1.4.15-0');
  assert.deepEqual(restored.records, [record]);
  assert.deepEqual(restored.images.get(imageFile), sourceImage);
});

test('older positive annotation records without newer optional fields remain importable', async () => {
  const legacy = { id: 'legacy', fileName: 'old.jpg', imageFile, boxes: record.boxes, createdAt: '2025-01-01T00:00:00Z' };
  const archive = await createCellTrainingDataBundle([legacy], () => sourceImage, '1.2.0');
  const restored = await parseCellTrainingDataBundle(archive);
  assert.equal(restored.records[0].sampleType, 'positive');
  assert.equal(restored.records[0].partial, undefined);
  assert.deepEqual(restored.records[0].missedBoxes, undefined);
});

test('legacy records without boxes are imported as true-background negative samples', async () => {
  const legacy = { id: 'legacy-negative', fileName: 'background.jpg', imageFile, boxes: [], createdAt: '2025-01-01T00:00:00Z' };
  const archive = await createCellTrainingDataBundle([legacy], () => sourceImage, '1.2.0');
  const restored = await parseCellTrainingDataBundle(archive);
  assert.equal(restored.records[0].sampleType, 'negative');
  assert.deepEqual(restored.records[0].boxes, []);
});

test('cell training backup rejects a source-image hash mismatch', async () => {
  const archive = await createCellTrainingDataBundle([record], () => sourceImage, '1.4.15-0');
  const tampered = Buffer.from(archive);
  // Parsing checks CRC and SHA-1; altering archive bytes must fail before records are applied.
  tampered[tampered.length - 10] ^= 0xff;
  await assert.rejects(() => parseCellTrainingDataBundle(tampered));
});


test('cell training backup carries the reusable .pt checkpoint for transfer-learning recovery', async () => {
  const checkpoint = Buffer.concat([Buffer.from([0x50, 0x4b, 0x03, 0x04]), Buffer.alloc(2048, 7)]);
  const archive = await createCellTrainingDataBundle([record], () => sourceImage, '1.4.15-0', () => checkpoint);
  const restored = await parseCellTrainingDataBundle(archive);
  assert.deepEqual(restored.checkpoint, checkpoint);
  assert.equal(restored.manifest.checkpoint.size, checkpoint.length);
});
