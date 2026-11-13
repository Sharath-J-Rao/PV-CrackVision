// v1.4.12.9: crack false-positive reduction - outer-ring edge definition from the detected cell grid + per-component agreement.
import test from 'node:test';
import assert from 'node:assert/strict';
import { cellGridBounds, isGridEdgeCell, isPerimeterCell } from './cellPriority.ts';
import { DEFAULT_TRAINING_SETTINGS, normalizeTrainingSettings } from './trainingSettings.ts';

// 6 x 4 grid of 40 px cells starting at (100, 100): grid outline is 240 x 160.
const cells = [];
for (let r = 0; r < 4; r++) for (let c = 0; c < 6; c++) cells.push({ left: 100 + c * 40, top: 100 + r * 40, width: 40, height: 40 });

test('grid bounds wrap the detected cells and skip malformed boxes', () => {
  assert.deepEqual(cellGridBounds(cells), { left: 100, top: 100, width: 240, height: 160 });
  assert.deepEqual(cellGridBounds([...cells, { left: NaN, top: 0, width: 5, height: 5 }, { left: 0, top: 0, width: 0, height: 5 }]), { left: 100, top: 100, width: 240, height: 160 });
  assert.equal(cellGridBounds([]), null);
});

test('only the outermost ring of the cell grid is an edge cell; the second ring is not', () => {
  const grid = cellGridBounds(cells);
  const edge = cells.filter((c) => isGridEdgeCell(c, grid));
  assert.equal(edge.length, 6 * 4 - 4 * 2); // 24 cells minus the 4 x 2 interior block = 16 outer-ring cells
  assert.equal(isGridEdgeCell({ left: 140, top: 140, width: 40, height: 40 }, grid), false, 'ring-2 cell');
  assert.equal(isGridEdgeCell({ left: 100, top: 140, width: 40, height: 40 }, grid), true, 'left column');
  assert.equal(isGridEdgeCell(cells[0], null), false);
});

test('the grid ring does not depend on how loose or tight the module region box is (the region-based ring does)', () => {
  const grid = cellGridBounds(cells);
  const outer = { left: 100, top: 140, width: 40, height: 40 }, ring2 = { left: 140, top: 140, width: 40, height: 40 };
  const looseRegion = { left: 50, top: 50, width: 340, height: 260 }; // frame + gap: the outer column is > 0.9 cell from the outline
  const tightRegion = { left: 120, top: 100, width: 200, height: 160 }; // cuts into the grid: ring 2 falls inside 0.9 cell of the outline
  assert.equal(isPerimeterCell(outer, looseRegion), false, 'region ring misses a real outer cell when the box is loose');
  assert.equal(isPerimeterCell(ring2, tightRegion), true, 'region ring pulls in ring 2 when the box is tight');
  assert.equal(isGridEdgeCell(outer, grid), true);
  assert.equal(isGridEdgeCell(ring2, grid), false);
});

test('edgeRequireAgreement defaults on, survives older saved settings, and respects an explicit off', () => {
  assert.equal(DEFAULT_TRAINING_SETTINGS.crackClassifier.edgeRequireAgreement, true);
  assert.equal(normalizeTrainingSettings({ crackClassifier: { edgeThresholdDrop: 0.1 } }).crackClassifier.edgeRequireAgreement, true);
  assert.equal(normalizeTrainingSettings({ crackClassifier: { edgeRequireAgreement: false } }).crackClassifier.edgeRequireAgreement, false);
  assert.equal(normalizeTrainingSettings({ crackClassifier: { edgeRequireAgreement: 'no' } }).crackClassifier.edgeRequireAgreement, true, 'non-boolean falls back to default');
});
