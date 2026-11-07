import test from 'node:test';
import assert from 'node:assert/strict';
import { gridFromYoloCellBoxes, applyYoloCellsToRegions } from './cellDetectionGeometry.ts';

const cells = [];
for (let row = 0; row < 2; row++) {
  for (let col = 0; col < 3; col++) {
    cells.push({ left: 20 + col * 42, top: 30 + row * 32, width: 36, height: 26, confidence: 0.9 });
  }
}

test('direct cell detections create actual per-cell rectangles and stable row/column IDs', () => {
  const grid = gridFromYoloCellBoxes(cells);
  assert.ok(grid);
  assert.equal(grid.method, 'yolo-cell-detection');
  assert.equal(grid.rows, 2);
  assert.equal(grid.columns, 3);
  assert.equal(grid.cells.length, 6);
  assert.deepEqual(grid.cells.map((cell) => [cell.row, cell.column]), [
    [1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3],
  ]);
});

test('direct cell detections replace classical geometry only in the matching region', () => {
  const regions = [
    { id: 'left', label: 'PV Array 1', isFallback: false, pixelBox: { left: 0, top: 0, width: 200, height: 120 } },
    { id: 'right', label: 'PV Array 2', isFallback: false, pixelBox: { left: 250, top: 0, width: 200, height: 120 } },
  ];
  const result = applyYoloCellsToRegions(regions, cells);
  assert.equal(result.applied, 6);
  assert.equal(result.regions[0].cellGrid?.cells.length, 6);
  assert.equal(result.regions[1].cellGrid, undefined);
});

test('whole-image fallback still accepts direct cell boxes without a trained module detector', () => {
  const regions = [{ id: 'fallback', label: 'Whole Image', isFallback: true, pixelBox: { left: 0, top: 0, width: 500, height: 300 } }];
  const result = applyYoloCellsToRegions(regions, cells);
  assert.equal(result.applied, 6);
  assert.equal(result.regions[0].cellGrid?.method, 'yolo-cell-detection');
});


import { suppressOversizedPatternOverlaps } from './cellDetectionGeometry.ts';

test('prefers repeated individual-cell boxes over one large enclosing detection', () => {
  const detections = [
    { left: 0, top: 0, width: 120, height: 70, confidence: 0.99 },
    { left: 12, top: 12, width: 22, height: 22, confidence: 0.82 },
    { left: 55, top: 12, width: 22, height: 22, confidence: 0.79 },
  ];
  const filtered = suppressOversizedPatternOverlaps(detections);
  assert.equal(filtered.length, 2);
  assert.ok(filtered.every((box) => box.width === 22 && box.height === 22));
});

test('still applies ordinary NMS to near-identical same-scale detections', () => {
  const detections = [
    { left: 10, top: 10, width: 30, height: 30, confidence: 0.95 },
    { left: 11, top: 10, width: 30, height: 30, confidence: 0.75 },
    { left: 60, top: 10, width: 30, height: 30, confidence: 0.8 },
  ];
  const filtered = suppressOversizedPatternOverlaps(detections);
  assert.equal(filtered.length, 2);
  assert.equal(filtered[0].confidence, 0.95);
});


test('does not reject a larger detection based on one isolated nested box', () => {
  const detections = [
    { left: 0, top: 0, width: 120, height: 70, confidence: 0.99 },
    { left: 12, top: 12, width: 22, height: 22, confidence: 0.82 },
  ];
  const filtered = suppressOversizedPatternOverlaps(detections);
  assert.equal(filtered.length, 2);
});
