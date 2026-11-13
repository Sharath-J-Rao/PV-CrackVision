import test from 'node:test';
import assert from 'node:assert/strict';
import { scaleCellBoxesToSourceImage } from './cellAnnotationCoordinates.ts';

test('maps annotation coordinates back to an original image after inference downscaling', () => {
  const boxes = [{ left: 100, top: 200, width: 40, height: 30 }];
  assert.deepEqual(
    scaleCellBoxesToSourceImage(boxes, 24000, 18000, 8000, 6000),
    [{ left: 300, top: 600, width: 120, height: 90 }],
  );
});

test('leaves coordinates unchanged when the inspected and source dimensions match', () => {
  const boxes = [{ left: 20, top: 30, width: 40, height: 30 }];
  assert.deepEqual(scaleCellBoxesToSourceImage(boxes, 1000, 800, 1000, 800), boxes);
});

test('rejects mismatched aspect ratios rather than silently shifting training labels', () => {
  assert.throws(() => scaleCellBoxesToSourceImage([], 2000, 1000, 1000, 1000), /different aspect ratios/);
});

test('rejects boxes outside the source image after scale conversion', () => {
  assert.throws(() => scaleCellBoxesToSourceImage([{ left: 900, top: 0, width: 200, height: 30 }], 1000, 800, 1000, 800), /outside the source image/);
});
