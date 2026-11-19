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


// ---- v1.4.9: the crack pass reuses the operator's Cell Annotation boxes -----------------------------------------------
import { normalizeReviewedCellBoxes } from './cellDetectionGeometry.ts';

const twoRegions = () => [
  { id: 'left', label: 'PV Array 1', isFallback: false, pixelBox: { left: 0, top: 0, width: 200, height: 120 }, cellGrid: { method: 'edge-map-periodicity', cells: [{ row: 1, column: 1, pixelBox: { left: 0, top: 0, width: 10, height: 10 }, confidence: 0.9, partial: false }], confidence: 0.9 } },
  { id: 'right', label: 'PV Array 2', isFallback: false, pixelBox: { left: 250, top: 0, width: 200, height: 120 }, cellGrid: { method: 'edge-map-periodicity', cells: [{ row: 1, column: 1, pixelBox: { left: 250, top: 0, width: 10, height: 10 }, confidence: 0.9, partial: false }], confidence: 0.9 } },
];

test('reviewed mode: classical grid is replaced where cells exist and CLEARED where the operator kept none', () => {
  const result = applyYoloCellsToRegions(twoRegions(), cells, { minCells: 1, clearUnassigned: true, adoptOrphans: true, imageWidth: 500, imageHeight: 300 });
  assert.equal(result.applied, 6);
  assert.equal(result.regions[0].cellGrid.method, 'yolo-cell-detection');
  assert.equal(result.regions[1].cellGrid, undefined, 'a region with no reviewed cells must not keep a guessed classical grid');
});

test('default (unreviewed YOLO) behaviour is unchanged: untouched regions keep their classical grid', () => {
  const result = applyYoloCellsToRegions(twoRegions(), cells);
  assert.equal(result.regions[1].cellGrid.method, 'edge-map-periodicity');
  assert.equal(result.orphans, 0);
});

test('reviewed mode keeps hand-drawn cells that fall outside every located panel in an extra region', () => {
  const outside = [{ left: 600, top: 40, width: 36, height: 26, confidence: 1 }, { left: 642, top: 40, width: 36, height: 26, confidence: 1 }];
  const result = applyYoloCellsToRegions(twoRegions(), [...cells, ...outside], { minCells: 1, clearUnassigned: true, adoptOrphans: true, imageWidth: 800, imageHeight: 300 });
  assert.equal(result.orphans, 2);
  assert.equal(result.applied, 8);
  const extra = result.regions.find((region) => region.id === 'REVIEWED-CELLS');
  assert.ok(extra);
  assert.equal(extra.cellGrid.cells.length, 2);
  assert.ok(extra.box.xmax <= 1 && extra.box.xmin >= 0);
});

test('without adoptOrphans, outside cells are dropped exactly as before', () => {
  const outside = [{ left: 600, top: 40, width: 36, height: 26, confidence: 1 }];
  const result = applyYoloCellsToRegions(twoRegions(), [...cells, ...outside]);
  assert.equal(result.applied, 6);
  assert.equal(result.regions.length, 2);
});

test('a small reviewed set (under 4 cells) is accepted in reviewed mode but not by the unreviewed rule', () => {
  const few = cells.slice(0, 2);
  assert.equal(applyYoloCellsToRegions(twoRegions(), few).applied, 0);
  assert.equal(applyYoloCellsToRegions(twoRegions(), few, { minCells: 1, clearUnassigned: true }).applied, 2);
  assert.equal(gridFromYoloCellBoxes(few), undefined);
  assert.equal(gridFromYoloCellBoxes(few, 1).cells.length, 2);
});

test('normalizeReviewedCellBoxes rescales to the inspected frame, clips, and drops malformed boxes without throwing', () => {
  const { boxes, dropped } = normalizeReviewedCellBoxes({
    boxes: [
      { left: 100, top: 50, width: 40, height: 30 },      // ok
      { left: 7990, top: 10, width: 40, height: 30 },     // pokes past the right edge -> clipped
      { left: 10, top: 10, width: 2, height: 30 },        // too narrow
      { left: Number.NaN, top: 10, width: 40, height: 30 }, // not finite
      null,                                                // garbage entry
    ],
    coordinateWidth: 4000, coordinateHeight: 3000,
  }, 8000, 6000);
  assert.equal(boxes.length, 2);
  assert.equal(dropped, 3);
  assert.deepEqual([boxes[0].left, boxes[0].top, boxes[0].width, boxes[0].height], [200, 100, 80, 60]);
  assert.ok(boxes[1].left + boxes[1].width <= 8000);
  assert.ok(boxes.every((box) => box.confidence === 1));
});

test('normalizeReviewedCellBoxes keeps coordinates when no source frame size is given', () => {
  const { boxes } = normalizeReviewedCellBoxes({ boxes: [{ left: 10, top: 20, width: 30, height: 40 }] }, 500, 500);
  assert.deepEqual([boxes[0].left, boxes[0].top, boxes[0].width, boxes[0].height], [10, 20, 30, 40]);
});
