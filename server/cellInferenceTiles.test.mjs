import test from 'node:test';
import assert from 'node:assert/strict';
import { buildCellInferenceTiles, offsetCellBoxes } from './cellInferenceTiles.ts';

test('no extra tiles are created for an image that fits the model tile size', () => {
  assert.deepEqual(buildCellInferenceTiles(1400, 1200, 960, 960), []);
});

test('large images get overlapping crops that cover all four edges', () => {
  const tiles = buildCellInferenceTiles(4000, 3000, 960, 960);
  assert.ok(tiles.length > 1);
  assert.equal(tiles[0].left, 0);
  assert.equal(tiles[0].top, 0);
  assert.ok(tiles.some((tile) => tile.left + tile.width === 4000));
  assert.ok(tiles.some((tile) => tile.top + tile.height === 3000));
  assert.ok(tiles.every((tile) => tile.width > 0 && tile.height > 0));
});

test('tiles are capped on very large images without leaving uncovered edges', () => {
  const tiles = buildCellInferenceTiles(24000, 16000, 960, 960, { maxTiles: 16 });
  assert.ok(tiles.length <= 16);
  assert.ok(tiles.some((tile) => tile.left + tile.width === 24000));
  assert.ok(tiles.some((tile) => tile.top + tile.height === 16000));
});

test('offsets tile-local boxes into the full-image coordinate system and clips safely', () => {
  const result = offsetCellBoxes(
    [{ left: 10, top: 20, width: 30, height: 40, confidence: 0.8 }],
    { left: 100, top: 200, width: 500, height: 500 },
    1000,
    1000,
  );
  assert.deepEqual(result, [{ left: 110, top: 220, width: 30, height: 40, confidence: 0.8 }]);
});
