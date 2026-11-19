import test from 'node:test';
import assert from 'node:assert/strict';
import { calculateLetterbox, DEFAULT_ONNX_IMAGE_SIZE, resolveOnnxInputShape } from './onnxInputShape.ts';

test('reads static 960x960 ONNX input dimensions instead of assuming 640', () => {
  assert.deepEqual(resolveOnnxInputShape({ name: 'images', isTensor: true, shape: [1, 3, 960, 960] }), {
    channels: 3, height: 960, width: 960,
  });
});

test('supports rectangular fixed ONNX model dimensions', () => {
  assert.deepEqual(resolveOnnxInputShape({ name: 'images', shape: [1, 3, 640, 960] }), {
    channels: 3, height: 640, width: 960,
  });
});

test('uses a safe 960 fallback for dynamic or unspecified dimensions', () => {
  assert.deepEqual(resolveOnnxInputShape({ name: 'images', shape: [1, 3, 'height', 'width'] }), {
    channels: 3, height: DEFAULT_ONNX_IMAGE_SIZE, width: DEFAULT_ONNX_IMAGE_SIZE,
  });
  assert.deepEqual(resolveOnnxInputShape(undefined), { channels: 3, height: 960, width: 960 });
});

test('rejects unsupported non-RGB model inputs clearly', () => {
  assert.throws(() => resolveOnnxInputShape({ shape: [1, 1, 960, 960] }), /expected 3-channel RGB/);
});

test('letterbox returns dimensions bounded by the model input and centered padding', () => {
  const result = calculateLetterbox(1920, 1080, 960, 960);
  assert.equal(result.scale, 0.5);
  assert.equal(result.resizedWidth, 960);
  assert.equal(result.resizedHeight, 540);
  assert.equal(result.padX, 0);
  assert.equal(result.padY, 210);
});
