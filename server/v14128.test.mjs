// v1.4.12.8: weighted box fusion, region growth to detected cells, trainer default migration, class-weight formula.
import test from 'node:test';
import assert from 'node:assert/strict';
import { fuseOverlappingCellBoxes, applyYoloCellsToRegions } from './cellDetectionGeometry.ts';
import { DEFAULT_TRAINING_SETTINGS, DEFAULTS_VERSION, upgradeLegacyDefaults } from './trainingSettings.ts';
import { computeClassWeights } from './trainingValidation.ts';

test('fusion averages overlapping duplicates, keeps max score, never chains onto neighbours', () => {
  const out = fuseOverlappingCellBoxes([
    { left: 100, top: 100, width: 50, height: 50, confidence: 0.9 },
    { left: 104, top: 100, width: 50, height: 50, confidence: 0.3 }, // same cell, other pass
    { left: 152, top: 100, width: 50, height: 50, confidence: 0.8 }, // neighbouring cell: must stay separate
  ], 0.5);
  assert.equal(out.length, 2);
  const fused = out.find((b) => b.left < 120);
  assert.equal(fused.confidence, 0.9);
  assert.ok(fused.left > 100 && fused.left < 104, 'edges are score-weighted, closer to the strong box');
  assert.equal(fuseOverlappingCellBoxes([], 0.5).length, 0);
  assert.equal(fuseOverlappingCellBoxes([{ left: NaN, top: 0, width: 5, height: 5, confidence: 1 }], 0.5).length, 1, 'malformed boxes pass through');
});

test('region box grows to contain its cells and never shrinks', () => {
  const region = { id: 'm1', label: 'Module 1', pixelBox: { left: 200, top: 200, width: 100, height: 100 }, box: { xmin: 0.2, ymin: 0.2, xmax: 0.3, ymax: 0.3 } };
  // centres 200 / 260 / 298 are inside the region; the outer cells stick out to x=180 and x=318
  const cells = [180, 240, 278].map((left) => ({ left, top: 250, width: 40, height: 40, confidence: 0.9 }));
  const grown = applyYoloCellsToRegions([region], cells, { minCells: 3, imageWidth: 1000, imageHeight: 1000, expandRegionToCells: true }).regions[0];
  assert.equal(grown.pixelBox.left, 180);
  assert.equal(grown.pixelBox.left + grown.pixelBox.width, 318);
  assert.equal(grown.pixelBox.top, 200, 'top edge not shrunk');
  assert.ok(Math.abs(grown.box.xmin - 0.18) < 1e-9);
  const plain = applyYoloCellsToRegions([region], cells, { minCells: 3 }).regions[0];
  assert.deepEqual(plain.pixelBox, region.pixelBox, 'default behaviour unchanged');
});

test('trainer defaults migrate only when still at the old default', () => {
  const old = structuredClone(DEFAULT_TRAINING_SETTINGS);
  Object.assign(old.cellDetector, { batch: 2, cosLr: false, flipud: 0, cache: false });
  old.cellDetector.batch = 2; delete old.defaultsVersion;
  const up = upgradeLegacyDefaults(old);
  assert.equal(up.cellDetector.batch, DEFAULT_TRAINING_SETTINGS.cellDetector.batch);
  assert.equal(up.cellDetector.cosLr, true);
  assert.equal(up.cellDetector.flipud, 0.5);
  assert.equal(up.cellDetector.cache, true);
  const edited = structuredClone(old); edited.cellDetector.batch = 6;
  assert.equal(upgradeLegacyDefaults(edited).cellDetector.batch, 6, 'operator value kept');
  assert.equal(DEFAULTS_VERSION, 3);
});

test('crack-classifier class weights equal N / (2 * class count)', () => {
  const samples = [...Array(715).fill({ y: 0 }), ...Array(70).fill({ y: 1 })];
  const { summary } = computeClassWeights(samples, true);
  assert.ok(Math.abs(summary.negativeWeight - 785 / (2 * 715)) < 1e-3);
  assert.ok(Math.abs(summary.positiveWeight - 785 / (2 * 70)) < 1e-3);
});

test('partial-annotation mask keeps marked cells + margin and greys the rest', async () => {
  const { buildAnnotatedMask, greyOutsideMask } = await import('./partialAnnotationMask.ts');
  const boxes = [{ left: 10, top: 10, width: 20, height: 20 }, { left: 32, top: 10, width: 20, height: 20 }]; // neighbours with a 2 px gridline gap
  const mask = buildAnnotatedMask(100, 60, boxes, 0.25); // margin = round(0.25 * 20) = 5
  const at = (x, y) => mask[y * 100 + x];
  assert.equal(at(20, 20), 1, 'inside a marked cell');
  assert.equal(at(31, 20), 1, 'gridline gap between marked cells stays visible');
  assert.equal(at(56, 20), 1, 'within margin of the last cell (52 + 5 = 57)');
  assert.equal(at(70, 20), 0, 'far unmarked area is greyed');
  assert.equal(at(20, 45), 0, 'unmarked row below is greyed');
  assert.equal(buildAnnotatedMask(10, 10, [{ left: NaN, top: 0, width: 5, height: 5 }], 0.2).every((v) => v === 0), true, 'malformed boxes ignored');
  const rgb = Buffer.alloc(100 * 60 * 3, 200);
  const out = greyOutsideMask(rgb, 100, 60, mask);
  assert.equal(out[(20 * 100 + 20) * 3], 200);
  assert.equal(out[(20 * 100 + 70) * 3], 114);
  assert.equal(rgb[(20 * 100 + 70) * 3], 200, 'input buffer is not mutated');
  assert.throws(() => greyOutsideMask(Buffer.alloc(5), 100, 60, mask));
});

test('partialMaskMargin is a validated setting with a default', async () => {
  assert.equal(DEFAULT_TRAINING_SETTINGS.cellDetector.partialMaskMargin, 0.15);
  const { normalizeTrainingSettings } = await import('./trainingSettings.ts');
  assert.equal(normalizeTrainingSettings({ cellDetector: { partialMaskMargin: 9 } }).cellDetector.partialMaskMargin, 1, 'clamped');
  assert.equal(normalizeTrainingSettings({}).cellDetector.partialMaskMargin, 0.15);
});
