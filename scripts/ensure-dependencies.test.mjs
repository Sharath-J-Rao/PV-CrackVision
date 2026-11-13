import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { findMissingDependencies } from './ensure-dependencies.mjs';

test('reports only declared packages that are absent from node_modules (scoped names included)', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'pvcv-deps-'));
  fs.writeFileSync(path.join(root, 'package.json'), JSON.stringify({ dependencies: { express: '1', '@scope/pkg': '1' }, devDependencies: { tsx: '1' } }));
  fs.mkdirSync(path.join(root, 'node_modules', 'express'), { recursive: true });
  fs.writeFileSync(path.join(root, 'node_modules', 'express', 'package.json'), '{}');
  assert.deepEqual(findMissingDependencies(root).sort(), ['@scope/pkg', 'tsx']);
});
