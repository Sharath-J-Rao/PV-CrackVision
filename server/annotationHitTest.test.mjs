import assert from 'node:assert/strict';
import test from 'node:test';
import { findAnnotationCellAtPoint, findValidatedCellAtPoint } from '../src/utils/annotationHitTest.ts';

const box = (left, top, width = 40, height = 24) => ({ left, top, width, height });
const region = (id, left, top) => ({
  id,
  label: id,
  pixelBox: box(left, top, 80, 48),
  cellGrid: {
    rows: 2, columns: 2, confidence: 0.9, pixelBox: box(left, top, 80, 48),
    cells: [
      { row: 1, column: 1, pixelBox: box(left, top), confidence: 0.9, partial: false },
      { row: 1, column: 2, pixelBox: box(left + 40, top), confidence: 0.9, partial: false },
      { row: 2, column: 1, pixelBox: box(left, top + 24), confidence: 0.9, partial: false },
      { row: 2, column: 2, pixelBox: box(left + 40, top + 24), confidence: 0.9, partial: false },
    ],
  },
});

test('hit-testing reaches cells in a second module without selecting it first', () => {
  const first = region('module-1', 10, 10);
  const second = region('module-2', 110, 10);
  const hit = findValidatedCellAtPoint([first, second], { left: 155, top: 18 });
  assert.equal(hit?.region.id, 'module-2');
  assert.equal(hit?.cell.row, 1);
  assert.equal(hit?.cell.column, 2);
});

test('gaps and modules without validated cells are not selectable', () => {
  const first = region('module-1', 10, 10);
  const second = { id: 'module-2', label: 'module-2', pixelBox: box(110, 10, 80, 48) };
  assert.equal(findValidatedCellAtPoint([first, second], { left: 95, top: 18 }), null);
  assert.equal(findValidatedCellAtPoint([second], { left: 120, top: 18 }), null);
});

test('an explicit reference-cell calibration overrides overlapping auto cells and preserves its drawn pitch', async () => {
  const { findCalibratedCellAtPoint } = await import('../src/utils/annotationHitTest.ts');
  const automatic = region('module-1', 10, 10);
  const calibrated = {
    id: 'cal-module-1',
    moduleId: 'module-1',
    label: 'module-1',
    moduleBox: box(10, 10, 80, 48),
    pixelBox: box(8, 8, 84, 52),
    rows: 2,
    columns: 2,
    confidence: 0.95,
    pitchWidth: 42,
    pitchHeight: 26,
    cells: [
      { row: 1, column: 1, pixelBox: box(8, 8, 42, 26), confidence: 0.95, partial: false },
      { row: 1, column: 2, pixelBox: box(50, 8, 42, 26), confidence: 0.95, partial: false },
      { row: 2, column: 1, pixelBox: box(8, 34, 42, 26), confidence: 0.95, partial: false },
      { row: 2, column: 2, pixelBox: box(50, 34, 42, 26), confidence: 0.95, partial: false },
    ],
  };
  const point = { left: 49, top: 20 }; // inside the calibrated 42px cell, outside the auto 40px cell
  const calibrationHit = findCalibratedCellAtPoint([calibrated], point);
  assert.equal(calibrationHit?.region.pitchWidth, 42);
  assert.equal(calibrationHit?.region.pitchHeight, 26);
  assert.deepEqual(calibrationHit?.cell.pixelBox, box(8, 8, 42, 26));
  assert.equal(findValidatedCellAtPoint([automatic], point)?.cell.column, 1);
});

test('the newest manual calibration wins when calibrated footprints overlap', async () => {
  const { findCalibratedCellAtPoint } = await import('../src/utils/annotationHitTest.ts');
  const make = (id, left) => ({
    id, label: id, moduleBox: box(left, 0, 40, 24), pixelBox: box(left, 0, 40, 24),
    rows: 1, columns: 1, confidence: 1, pitchWidth: 40, pitchHeight: 24,
    cells: [{ row: 1, column: 1, pixelBox: box(left, 0, 40, 24), confidence: 1, partial: false }],
  });
  const hit = findCalibratedCellAtPoint([make('older', 10), make('newer', 10)], { left: 20, top: 10 });
  assert.equal(hit?.region.id, 'newer');
});


test('calibrated pitch is the only hit target in its module; automatic detection remains active in other modules', async () => {
  const automaticOne = region('module-1', 10, 10);
  const automaticTwo = region('module-2', 110, 10);
  const calibrated = {
    id: 'cal-module-1', moduleId: 'module-1', label: 'module-1',
    moduleBox: box(10, 10, 80, 48), pixelBox: box(8, 8, 84, 52),
    rows: 1, columns: 1, confidence: 0.95, pitchWidth: 42, pitchHeight: 26,
    cells: [
      { row: 1, column: 1, pixelBox: box(8, 8, 42, 26), confidence: 0.95, partial: false },
    ],
  };

  // This is in the old automatic module's second cell but outside the manually drawn
  // calibrated lattice. It must not silently select an obsolete auto rectangle.
  assert.equal(findAnnotationCellAtPoint([automaticOne, automaticTwo], [calibrated], { left: 70, top: 18 }), null);

  const calibratedHit = findAnnotationCellAtPoint([automaticOne, automaticTwo], [calibrated], { left: 49, top: 20 });
  assert.equal(calibratedHit?.source, 'calibrated');
  assert.equal(calibratedHit?.cell.pixelBox.width, 42);
  assert.equal(calibratedHit?.cell.pixelBox.height, 26);

  const otherModuleHit = findAnnotationCellAtPoint([automaticOne, automaticTwo], [calibrated], { left: 155, top: 18 });
  assert.equal(otherModuleHit?.source, 'automatic');
  assert.equal(otherModuleHit?.region.id, 'module-2');
});
