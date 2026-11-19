import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// The built-in classical ridge detector is opt-in (default OFF). These checks pin the whole chain
// (type -> UI default -> UI control -> server parser -> pipeline gate) so a refactor cannot silently re-enable it.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = (p) => fs.readFileSync(path.join(root, p), 'utf8');

test('classical ridge detector: config flag exists and defaults to OFF in the UI', () => {
  assert.match(read('src/types/inspection.ts'), /enableClassicalRidge:\s*boolean;/);
  assert.match(read('src/App.tsx'), /enableClassicalRidge:\s*false/);
});

test('classical ridge detector: settings checkbox is bound to the flag', () => {
  const app = read('src/App.tsx');
  assert.match(app, /checked=\{config\.enableClassicalRidge\}/);
  assert.match(app, /enableClassicalRidge:\s*e\.target\.checked/);
});

test('classical ridge detector: server only enables it for a literal true (missing / garbled / legacy header stays OFF)', () => {
  assert.match(read('server.ts'), /enableClassicalRidge:\s*obj\.enableClassicalRidge === true/);
});

test('classical ridge detector: pipeline never calls it unless the flag is set', () => {
  const src = read('server/cvPipeline.ts');
  const calls = src.match(/detectClassicalModuleSpiderWebs\(/g) ?? [];
  assert.equal(calls.length, 2, 'one definition + exactly one gated call site');
  assert.match(src, /config\.enableClassicalRidge\s*\?\s*detectClassicalModuleSpiderWebs\(/);
});
