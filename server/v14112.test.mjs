import test from 'node:test';
import assert from 'node:assert/strict';
import { applyGroundGate } from './cvPipeline.ts';

// 10x10 colour mask over a 100x100 image: all soil, no silicon (a "dark/dirty" or mis-masked area).
const mask = (v) => new Uint8Array(100).fill(v);
const cell = { pixelBox: { left: 10, top: 10, width: 30, height: 30 } };
const stage1 = (withCells) => ({
  scaleWidth: 10, scaleHeight: 10, soilMask: mask(1), siliconMask: mask(0),
  moduleRegions: [{ id: 'm', isFallback: false, pixelBox: { left: 0, top: 0, width: 100, height: 100 }, ...(withCells ? { cellGrid: { cells: [cell] } } : {}) }],
});
const box = { left: 12, top: 12, width: 20, height: 20 };

test('soil-masked box is still rejected without a confirmed cell (ground protection unchanged)', () => {
  assert.equal(applyGroundGate(box, stage1(true), 100, 100, false), null);
  assert.equal(applyGroundGate(box, stage1(false), 100, 100, true), null);
});
test('detection inside a confirmed cell bypasses the colour mask', () => {
  assert.deepEqual(applyGroundGate(box, stage1(true), 100, 100, true), box);
});
test('detection outside every confirmed cell is still gated', () => {
  assert.equal(applyGroundGate({ left: 60, top: 60, width: 20, height: 20 }, stage1(true), 100, 100, true), null);
});
