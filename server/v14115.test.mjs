import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import sharp from 'sharp';

process.env.PV_CRACKVISION_DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'pv-missed-'));
const { findMissedCells } = await import('../src/utils/missedCells.ts');
const { saveCellAnnotations } = await import('./cellAnnotationStore.ts');
const { createDataset } = await import('./cellTraining.ts');
const { CELL_YOLO_DIR } = await import('./dataPaths.ts');
const { DEFAULT_TRAINING_SETTINGS, normalizeTrainingSettings } = await import('./trainingSettings.ts').then((m) => ({ DEFAULT_TRAINING_SETTINGS: m.getTrainingSettings(), normalizeTrainingSettings: m.normalizeTrainingSettings }));

const box = (left, top, w = 40, h = 40) => ({ left, top, width: w, height: h });

test('boxes without an overlapping model detection are flagged as missed', () => {
  const detected = [box(10, 10), box(100, 10)];
  const reviewed = [box(12, 11), box(100, 10), box(300, 300)]; // last one is operator-added
  assert.deepEqual(findMissedCells(reviewed, detected), [box(300, 300)]);
  assert.deepEqual(findMissedCells(reviewed, undefined), [], 'no model output -> nothing attributed to the model');
  assert.equal(findMissedCells(reviewed, []).length, 3, 'model returned nothing -> every cell was missed');
});

test('missedCellRepeat defaults to 2 and is clamped to 0..10', () => {
  assert.equal(DEFAULT_TRAINING_SETTINGS.cellDetector.missedCellRepeat, 2);
  assert.equal(normalizeTrainingSettings({ cellDetector: { missedCellRepeat: 99 } }).cellDetector.missedCellRepeat, 10);
  assert.equal(normalizeTrainingSettings({ cellDetector: { missedCellRepeat: -3 } }).cellDetector.missedCellRepeat, 0);
});

test('images with missed cells are repeated in TRAIN only; validation is never oversampled', async () => {
  for (let i = 0; i < 6; i++) {
    const png = await sharp({ create: { width: 200, height: 200, channels: 3, background: { r: 20 + i * 30, g: 60, b: 90 } } }).png().toBuffer();
    saveCellAnnotations(png, `img${i}.png`, [box(10, 10), box(60, 10), box(110, 10)], undefined, i % 2 === 0 ? [box(110, 10)] : []);
  }
  const summary = await createDataset();
  const dir = path.join(CELL_YOLO_DIR, 'dataset');
  const find = (split) => fs.readdirSync(path.join(dir, 'images', split));
  assert.ok(find('train').some((f) => /_m1\.jpg$/.test(f)) && find('train').some((f) => /_m2\.jpg$/.test(f)));
  assert.ok(!find('val').some((f) => /_m\d+\.jpg$/.test(f)), 'validation split must not be oversampled');
  assert.ok(summary.missedCellImages >= 1 && summary.missedCopies >= 2);
  for (const f of find('train').filter((n) => /_m\d\.jpg$/.test(n))) assert.ok(fs.existsSync(path.join(dir, 'labels', 'train', f.replace('.jpg', '.txt'))));
});
