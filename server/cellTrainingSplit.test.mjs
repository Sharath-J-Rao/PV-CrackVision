import test from 'node:test';
import assert from 'node:assert/strict';
import { chooseValidationImageFiles } from './cellTrainingSplit.ts';

test('validation subset balances box counts instead of assigning the largest image to validation', () => {
  const records = [
    { imageFile: '6_5GSD_DJI_20260925180926_0293_Z.jpg', boxes: Array.from({ length: 2 }) },
    { imageFile: '6_5GSD_DJI_20260925180920_0291_Z.jpg', boxes: Array.from({ length: 13 }) },
    { imageFile: '6_5GSD_DJI_20260925180917_0290_Z.jpg', boxes: Array.from({ length: 78 }) },
  ];
  assert.deepEqual([...chooseValidationImageFiles(records)], ['6_5GSD_DJI_20260925180920_0291_Z.jpg']);
});

test('split is deterministic and leaves at least one training image', () => {
  const records = [1, 2, 3, 4, 5].map((count) => ({ imageFile: `img-${count}.jpg`, boxes: Array.from({ length: count }) }));
  const first = chooseValidationImageFiles(records);
  const second = chooseValidationImageFiles([...records].reverse());
  assert.deepEqual([...first], [...second]);
  assert.equal(first.size, 1);
});
