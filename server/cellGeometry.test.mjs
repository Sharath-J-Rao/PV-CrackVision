import test from 'node:test';
import assert from 'node:assert/strict';
import { buildValidatedCellGeometry } from './cellGeometry.ts';

function fillRect(mask, width, x0, y0, x1, y1) {
  for (let y = Math.max(0, y0); y < Math.min(mask.length / width, y1); y++) {
    for (let x = Math.max(0, x0); x < Math.min(width, x1); x++) mask[y * width + x] = 1;
  }
}

test('emits individual cells and leaves inter-array gaps empty', () => {
  const width = 140, height = 60;
  const mask = new Uint8Array(width * height);
  // Two adjacent visible cell-fields, six columns each, separated by a two-cell gap.
  for (let row = 0; row < 3; row++) {
    for (const [first, last] of [[0, 6], [8, 14]]) {
      for (let col = first; col < last; col++) {
        fillRect(mask, width, col * 10 + 1, row * 20 + 1, (col + 1) * 10 - 1, (row + 1) * 20 - 1);
      }
    }
  }
  const result = buildValidatedCellGeometry({
    gridX0: 0, gridY0: 0, pitchX: 10, pitchY: 20, rows: 3, columns: 14,
    width, height, imageWidth: width * 2, imageHeight: height * 2,
    pvSiliconMask: mask, periodicityConfidence: 0.9,
    clipBounds: { left: 0, top: 0, right: width, bottom: height },
  });
  assert.equal(result.cells.length, 36);
  assert.equal(result.cells.some((cell) => cell.column === 7 || cell.column === 8), false);
  assert.equal(result.cells.filter((cell) => cell.row === 2).length, 12);
  assert.ok(result.confidence > 0.8);
});

test('retains clipped partial cells at image boundaries', () => {
  const width = 25, height = 45;
  const mask = new Uint8Array(width * height);
  mask.fill(1);
  const result = buildValidatedCellGeometry({
    gridX0: -5, gridY0: -10, pitchX: 10, pitchY: 20, rows: 3, columns: 3,
    width, height, imageWidth: width, imageHeight: height,
    pvSiliconMask: mask, periodicityConfidence: 0.85,
    clipBounds: { left: 0, top: 0, right: width, bottom: height },
  });
  assert.ok(result.cells.some((cell) => cell.partial));
  assert.ok(result.cells.every((cell) => cell.pixelBox.left >= 0 && cell.pixelBox.top >= 0));
  assert.ok(result.cells.every((cell) => cell.pixelBox.width > 0 && cell.pixelBox.height > 0));
});

test('does not fabricate cells on a non-PV gap or ground mask', () => {
  const result = buildValidatedCellGeometry({
    gridX0: 0, gridY0: 0, pitchX: 10, pitchY: 20, rows: 3, columns: 10,
    width: 100, height: 60, imageWidth: 100, imageHeight: 60,
    pvSiliconMask: new Uint8Array(100 * 60), periodicityConfidence: 0.95,
    clipBounds: { left: 0, top: 0, right: 100, bottom: 60 },
  });
  assert.equal(result.cells.length, 0);
  assert.equal(result.confidence, 0);
});


test('keeps a full cell crossing a band boundary instead of truncating its rectangle', () => {
  const width = 40, height = 80;
  const mask = new Uint8Array(width * height);
  // The complete PV cell extends above the current band ROI; its center/area still
  // intersects the band, so it must be kept as a full rectangle, not clipped at y=50.
  fillRect(mask, width, 0, 40, 40, 80);
  const result = buildValidatedCellGeometry({
    gridX0: 0, gridY0: 40, pitchX: 10, pitchY: 20, rows: 2, columns: 4,
    width, height, imageWidth: width, imageHeight: height,
    pvSiliconMask: mask, periodicityConfidence: 0.9,
    clipBounds: { left: 0, top: 50, right: 40, bottom: 80 },
  });
  assert.ok(result.cells.length >= 4);
  assert.ok(result.cells.every((cell) => cell.pixelBox.height === 20));
  assert.ok(result.cells.some((cell) => cell.pixelBox.top === 40));
});


test('rejects a PV-coloured shadow slot when its outer cell seam is missing', () => {
  const width = 50, height = 50;
  const mask = new Uint8Array(width * height);
  const vertical = new Float32Array(width * height);
  const horizontal = new Float32Array(width * height);
  // Three real columns x two rows. A fourth dark slot is PV-coloured but has no
  // right-hand cell seam and must not be emitted as a cell.
  fillRect(mask, width, 2, 2, 42, 42);
  for (const x of [2, 12, 22, 32]) {
    for (let y = 2; y < 42; y++) vertical[y * width + x] = 250;
  }
  for (const y of [2, 22, 42]) {
    for (let x = 2; x < 32; x++) horizontal[y * width + x] = 250;
  }
  const result = buildValidatedCellGeometry({
    gridX0: 2, gridY0: 2, pitchX: 10, pitchY: 20, rows: 2, columns: 4,
    width, height, imageWidth: width, imageHeight: height,
    pvSiliconMask: mask, verticalEdgeMap: vertical, horizontalEdgeMap: horizontal,
    periodicityConfidence: 0.95,
    clipBounds: { left: 2, top: 2, right: 42, bottom: 42 },
  });
  assert.equal(result.cells.length, 6);
  assert.equal(result.cells.some((cell) => cell.column === 4), false);
});
