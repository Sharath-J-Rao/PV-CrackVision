import test from 'node:test';
import assert from 'node:assert/strict';
import { isSupportedNodeVersion } from '../scripts/node-runtime.mjs';

test('Node runtime gate matches Vite 8 supported engine range', () => {
  assert.equal(isSupportedNodeVersion('20.18.1'), false);
  assert.equal(isSupportedNodeVersion('20.19.0'), true);
  assert.equal(isSupportedNodeVersion('22.11.0'), false);
  assert.equal(isSupportedNodeVersion('22.12.0'), true);
  assert.equal(isSupportedNodeVersion('23.0.0'), true);
  assert.equal(isSupportedNodeVersion('24.1.0'), true);
  assert.equal(isSupportedNodeVersion('21.7.0'), false);
  assert.equal(isSupportedNodeVersion('not-a-version'), false);
});
