// v1.4.12.7 regression tests: small-fragment suppression, robust typical-cell size, legacy-default upgrade.
import test from 'node:test';
import assert from 'node:assert/strict';
import { filterCellsByType, referenceCell, suppressOverlappingFragments } from './cellTypeFilter.ts';
import { DEFAULT_TRAINING_SETTINGS, DEFAULTS_VERSION, upgradeLegacyDefaults } from './trainingSettings.ts';

const cell = (col, row, confidence = 0.8) => ({ left: col * 42, top: row * 52, width: 40, height: 50, confidence });
const grid = (cols, rows) => Array.from({ length: cols * rows }, (_, i) => cell(i % cols, Math.floor(i / cols)));

test('fragments inside a cell and shifted duplicates are removed; real cells and their order are kept', () => {
  const cells = grid(6, 4);
  const fragment = { left: cells[7].left + 5, top: cells[7].top + 5, width: 14, height: 16, confidence: 0.95 }; // MORE confident than its cell
  const shifted = { ...cells[3], left: cells[3].left + 14, confidence: 0.6 };                                  // same size, 65 % overlap
  const input = [fragment, ...cells, shifted];
  const { kept, rejected } = suppressOverlappingFragments(input, 0.5);
  assert.deepEqual(rejected, [fragment, shifted]);
  assert.deepEqual(kept, cells, 'all 24 real cells survive in their original order');
});

test('touching neighbours are never treated as fragments, and 0 disables the stage', () => {
  const cells = grid(5, 3);
  assert.equal(suppressOverlappingFragments(cells, 0.5).rejected.length, 0);
  const fragment = { left: 3, top: 3, width: 10, height: 10, confidence: 0.9 };
  assert.equal(suppressOverlappingFragments([fragment, ...cells], 0).rejected.length, 0);
});

test('typical cell size follows the dominant cluster even when small fragments outnumber the confident cells', () => {
  const real = Array.from({ length: 14 }, (_, i) => ({ left: i * 50, top: 0, width: 40, height: 50, confidence: 0.6 }));
  const junk = Array.from({ length: 18 }, (_, i) => ({ left: i * 20, top: 200, width: 8 + i, height: 10 + i, confidence: 0.7 })); // small, varied and MORE confident
  const ref = referenceCell([...real, ...junk]);
  assert.ok(ref.area > 1500, `reference area ${ref.area} should be a real cell (2000), not a fragment (168)`);
});

test('default filter drops tiny detections around real cells (end to end)', () => {
  const cells = grid(8, 5);
  const tiny = cells.slice(0, 10).map((c) => ({ left: c.left + 8, top: c.top + 9, width: 12, height: 14, confidence: 0.9 }));
  const { kept, rejected } = filterCellsByType([...tiny, ...cells], DEFAULT_TRAINING_SETTINGS.cellTypeFilter);
  assert.equal(kept.length, cells.length);
  assert.equal(rejected.length, tiny.length);
});

test('older settings files are upgraded once; operator-edited values are kept', () => {
  const legacy = structuredClone(DEFAULT_TRAINING_SETTINGS);
  Object.assign(legacy.cellInference, { maxCandidatesBeforeNms: 700, maxDecoderBoxes: 1200, maxFinalBoxes: 1600 });
  Object.assign(legacy.cellTypeFilter, { minAreaRelative: 0.55, minShortRelative: 0.6, minConfidence: 0.05 });
  delete legacy.defaultsVersion;
  legacy.cellTypeFilter.minShortRelative = 0.8; // edited by the operator
  const upgraded = upgradeLegacyDefaults(legacy);
  assert.equal(upgraded.cellInference.maxCandidatesBeforeNms, DEFAULT_TRAINING_SETTINGS.cellInference.maxCandidatesBeforeNms);
  assert.equal(upgraded.cellTypeFilter.minAreaRelative, DEFAULT_TRAINING_SETTINGS.cellTypeFilter.minAreaRelative);
  assert.equal(upgraded.cellTypeFilter.minConfidence, DEFAULT_TRAINING_SETTINGS.cellTypeFilter.minConfidence);
  assert.equal(upgraded.cellTypeFilter.minShortRelative, 0.8, 'an edited value must survive');
  assert.equal(legacy.cellInference.maxFinalBoxes, 1600, 'input is not mutated');
  const current = { ...legacy, defaultsVersion: DEFAULTS_VERSION };
  assert.equal(upgradeLegacyDefaults(current), current, 'already upgraded files are untouched');
});
