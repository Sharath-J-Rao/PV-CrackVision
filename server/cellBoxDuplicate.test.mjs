import test from 'node:test';
import assert from 'node:assert/strict';
import { estimateCellGaps, snapDuplicateOffset, duplicateStep } from '../src/utils/cellBoxPlacement.ts';

const row = [0, 1, 2, 3].map((i) => ({ left: i * 52, top: 0, width: 50, height: 50 })); // 2px seam
const bounds = { width: 400, height: 300 };

test('estimates seam from neighbours', () => {
  const g = estimateCellGaps(row);
  assert.equal(g.x, 2);
});
test('snaps a rough drag to one exact cell pitch and locks the other axis', () => {
  const d = snapDuplicateOffset({ x: 58, y: 6 }, [row[0]], { x: 2, y: 2 });
  assert.deepEqual(d, { x: 52, y: 0 });
});
test('duplicateStep moves one pitch, or returns null with no room', () => {
  const copy = duplicateStep([row[3]], 'right', { x: 2, y: 2 }, bounds);
  assert.equal(copy[0].left, 156 + 52);
  assert.equal(duplicateStep([row[0]], 'left', { x: 2, y: 2 }, bounds), null);
  assert.equal(duplicateStep(row, 'right', { x: 2, y: 2 }, { width: 500, height: 300 })[0].left, 208); // whole row steps by its width + seam
  assert.equal(duplicateStep(row, 'right', { x: 2, y: 2 }, bounds), null); // 208+206 > 400
});

import { duplicateRepeat } from '../src/utils/cellBoxPlacement.ts';
test('repeat fills to the image edge, honours a count, and skips occupied slots', () => {
  const g = { x: 2, y: 2 };
  const toEdge = duplicateRepeat([row[0]], 'right', g, { width: 400, height: 300 }, 0);
  assert.equal(toEdge.length, 6); // lefts 52..312, next (364) would end at 414 > 400
  assert.equal(duplicateRepeat([row[0]], 'right', g, { width: 400, height: 300 }, 3).length, 3);
  assert.equal(duplicateRepeat([row[0]], 'right', g, { width: 400, height: 300 }, 0, row).length, 3); // slots 1-3 already taken
});
