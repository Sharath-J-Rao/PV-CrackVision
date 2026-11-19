import test from 'node:test';
import assert from 'node:assert/strict';
import { isMissedCrack } from '../src/utils/missedCells.ts';
import { getTrainingSettings, normalizeTrainingSettings } from './trainingSettings.ts';

const cell = { left: 100, top: 100, width: 50, height: 50 };
const det = (x, y, w = 10, h = 10) => ({ pointPx: { x, y }, pixelBox: { left: x - w / 2, top: y - h / 2, width: w, height: h } });

test('a cracked cell with no model detection inside it is a missed crack', () => {
  assert.equal(isMissedCrack(cell, []), true);
  assert.equal(isMissedCrack(cell, [det(400, 400)]), true);
  assert.equal(isMissedCrack(cell, [det(120, 120)]), false, 'model point inside the cell');
  assert.equal(isMissedCrack(cell, [{ pointPx: { x: 10, y: 10 }, pixelBox: { left: 110, top: 110, width: 20, height: 20 } }]), false, 'model box centre inside the cell');
});

test('missedCrackWeight defaults to 3 and is clamped to 1..10', () => {
  assert.equal(getTrainingSettings().crackClassifier.missedCrackWeight, 3);
  assert.equal(normalizeTrainingSettings({ crackClassifier: { missedCrackWeight: 99 } }).crackClassifier.missedCrackWeight, 10);
  assert.equal(normalizeTrainingSettings({ crackClassifier: { missedCrackWeight: 0 } }).crackClassifier.missedCrackWeight, 1);
});
