import test from 'node:test';
import assert from 'node:assert/strict';
import { buildSelectableCellRegions } from '../src/utils/cellAnnotationTargets.ts';
import { findAnnotationCellAtPoint } from '../src/utils/annotationHitTest.ts';

const region = (id, left, top, width = 200, height = 120, isFallback = false) => ({
  id,
  label: id,
  isFallback,
  pixelBox: { left, top, width, height },
  box: { xmin: left / 1000, ymin: top / 1000, xmax: (left + width) / 1000, ymax: (top + height) / 1000 },
});

test('a single visible YOLO box becomes selectable even when detector returns fewer than four boxes', () => {
  const source = [region('module-1', 0, 0)];
  const boxes = [{ left: 20, top: 30, width: 32, height: 26 }];
  const result = buildSelectableCellRegions(source, boxes, 1000, 1000);
  assert.equal(result[0].cellGrid?.method, 'yolo-cell-detection');
  assert.equal(result[0].cellGrid?.cells.length, 1);
  assert.deepEqual(result[0].cellGrid?.cells[0].pixelBox, boxes[0]);
});

test('YOLO hit targets replace stale classical geometry in the same module', () => {
  const source = [{
    ...region('module-1', 0, 0),
    cellGrid: {
      rows: 2, columns: 2, pixelBox: { left: 0, top: 0, width: 100, height: 80 },
      cells: [{ row: 1, column: 1, pixelBox: { left: 1, top: 1, width: 10, height: 10 }, confidence: 0.9, partial: false }],
      pitchX: 50, pitchY: 40, confidence: 0.9, method: 'edge-map-periodicity',
    },
  }];
  const boxes = [
    { left: 25, top: 25, width: 30, height: 24 },
    { left: 70, top: 25, width: 30, height: 24 },
  ];
  const result = buildSelectableCellRegions(source, boxes, 1000, 1000);
  assert.equal(result[0].cellGrid?.method, 'yolo-cell-detection');
  assert.equal(result[0].cellGrid?.cells.length, 2);
  assert.deepEqual(result[0].cellGrid?.cells.map((cell) => [cell.row, cell.column]), [[1, 1], [1, 2]]);
});

test('direct YOLO boxes are assigned to their containing module rather than the first module', () => {
  const source = [region('left', 0, 0), region('right', 300, 0)];
  const boxes = [{ left: 340, top: 35, width: 30, height: 24 }];
  const result = buildSelectableCellRegions(source, boxes, 1000, 1000);
  assert.equal(result[0].cellGrid, undefined);
  assert.equal(result[1].cellGrid?.cells.length, 1);
});

test('unassigned visible detections remain selectable in a review region', () => {
  const source = [region('module-1', 0, 0, 100, 80)];
  const boxes = [{ left: 500, top: 500, width: 30, height: 24 }];
  const result = buildSelectableCellRegions(source, boxes, 1000, 1000);
  const review = result.find((item) => item.id === 'yolo-unassigned-cells');
  assert.ok(review);
  assert.equal(review.cellGrid?.cells.length, 1);
});

test('no direct detections preserves the existing classical cell grid', () => {
  const source = [{
    ...region('module-1', 0, 0),
    cellGrid: {
      rows: 1, columns: 1, pixelBox: { left: 10, top: 10, width: 20, height: 20 },
      cells: [{ row: 1, column: 1, pixelBox: { left: 10, top: 10, width: 20, height: 20 }, confidence: 0.8, partial: false }],
      pitchX: 20, pitchY: 20, confidence: 0.8, method: 'edge-map-periodicity',
    },
  }];
  const result = buildSelectableCellRegions(source, [], 1000, 1000);
  assert.equal(result[0].cellGrid?.method, 'edge-map-periodicity');
});


test('the exact visible YOLO box is returned by the same annotation hit tester used by canvas clicks', () => {
  const source = [region('module-1', 0, 0)];
  const boxes = [{ left: 40, top: 35, width: 30, height: 24 }];
  const selectable = buildSelectableCellRegions(source, boxes, 1000, 1000);
  const hit = findAnnotationCellAtPoint(selectable, [], { left: 55, top: 47 });
  assert.equal(hit?.source, 'automatic');
  assert.equal(hit?.region.id, 'module-1');
  assert.deepEqual(hit?.cell.pixelBox, boxes[0]);
});
