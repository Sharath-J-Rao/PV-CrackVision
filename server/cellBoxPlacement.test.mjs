import test from 'node:test';
import assert from 'node:assert/strict';
import { isCellBoxPlacementCommitKey, placeCellBoxesAt, translateCellBoxes } from '../src/utils/cellBoxPlacement.ts';

const bounds = { width: 1000, height: 800 };

test('paste places one copied cell box at the chosen destination center', () => {
  const [placed] = placeCellBoxesAt([{ left: 10, top: 20, width: 40, height: 30 }], { x: 500, y: 400 }, bounds);
  assert.deepEqual(placed, { left: 480, top: 385, width: 40, height: 30 });
});

test('paste preserves copied group spacing and dimensions', () => {
  const pasted = placeCellBoxesAt([
    { left: 10, top: 20, width: 40, height: 30 },
    { left: 60, top: 20, width: 40, height: 30 },
  ], { x: 500, y: 400 }, bounds);
  assert.deepEqual(pasted, [
    { left: 455, top: 385, width: 40, height: 30 },
    { left: 505, top: 385, width: 40, height: 30 },
  ]);
});

test('moving a selected group preserves spacing and clamps the whole group in-image', () => {
  const moved = translateCellBoxes([
    { left: 10, top: 20, width: 40, height: 30 },
    { left: 60, top: 20, width: 40, height: 30 },
  ], { x: -100, y: 50 }, bounds);
  assert.deepEqual(moved, [
    { left: 0, top: 70, width: 40, height: 30 },
    { left: 50, top: 70, width: 40, height: 30 },
  ]);
});

test('paste clamps the complete copied group without clipping any box', () => {
  const pasted = placeCellBoxesAt([
    { left: 10, top: 20, width: 40, height: 30 },
    { left: 60, top: 20, width: 40, height: 30 },
  ], { x: 2000, y: 2000 }, bounds);
  assert.deepEqual(pasted, [
    { left: 910, top: 770, width: 40, height: 30 },
    { left: 960, top: 770, width: 40, height: 30 },
  ]);
});

test('Space commits an active copied-box placement without treating other keys as commit', () => {
  assert.equal(isCellBoxPlacementCommitKey(' ', 'Space'), true);
  assert.equal(isCellBoxPlacementCommitKey('Spacebar', 'Space'), true);
  assert.equal(isCellBoxPlacementCommitKey('Escape', 'Escape'), false);
  assert.equal(isCellBoxPlacementCommitKey('Enter', 'Enter'), false);
});
