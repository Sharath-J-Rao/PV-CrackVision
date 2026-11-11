import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { framesBlockingCrackStage } from '../src/utils/crackGate.ts';
import { splitHoldoutByImage } from './trainingValidation.ts';
import { checkReleaseVersion } from '../scripts/check-release-version.mjs';
import { dirSize } from './dataFolder.ts';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const frame = (id, status) => ({ id, imageSrc: 'x', cellDetection: status ? { status } : undefined });

test('single-image crack run is blocked only by its own target', () => {
  const frames = [frame('a', 'completed'), frame('b', undefined)];
  assert.equal(framesBlockingCrackStage(frames, [frames[0]]).length, 0);
  assert.deepEqual(framesBlockingCrackStage(frames, [frames[1]]).map((f) => f.id), ['b']);
  assert.deepEqual(framesBlockingCrackStage(frames).map((f) => f.id), ['b'], 'all-images action still requires every frame');
});

test('holdout split keeps whole images apart and flags the shared fallback', () => {
  const mk = (img, y) => ({ imageFile: img, y });
  const four = [mk('a', 1), mk('a', 0), mk('b', 1), mk('b', 0), mk('c', 1), mk('c', 0), mk('d', 1), mk('d', 0)];
  const split = splitHoldoutByImage(four);
  assert.equal(split.independent, true);
  const sel = new Set(split.selection.map((s) => s.imageFile));
  assert.ok(split.test.every((s) => !sel.has(s.imageFile)), 'no image may appear in both sets');
  const small = splitHoldoutByImage(four.slice(0, 4));
  assert.equal(small.independent, false);
  assert.equal(small.test, small.selection);
});

test('release identity of this source tree is consistent', () => {
  const { errors } = checkReleaseVersion(root);
  assert.deepEqual(errors, []);
});

test('release check detects a stale publish script', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pvrel-'));
  for (const f of ['package.json', 'package-lock.json', 'metadata.json', 'CHANGELOG.md']) fs.copyFileSync(path.join(root, f), path.join(tmp, f));
  fs.mkdirSync(path.join(tmp, '.github/workflows'), { recursive: true });
  fs.copyFileSync(path.join(root, '.github/workflows/windows-desktop.yml'), path.join(tmp, '.github/workflows/windows-desktop.yml'));
  fs.writeFileSync(path.join(tmp, 'Publish_v9.9.9_Desktop.ps1'), '');
  assert.ok(checkReleaseVersion(tmp).errors.length >= 2);
  fs.rmSync(tmp, { recursive: true, force: true });
});

test('async dirSize totals files and honours the truncation limit', async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pvsize-'));
  fs.mkdirSync(path.join(tmp, 'sub'));
  for (let i = 0; i < 5; i++) fs.writeFileSync(path.join(tmp, i % 2 ? 'sub' : '.', `f${i}.bin`), Buffer.alloc(10));
  assert.deepEqual(await dirSize(tmp), { bytes: 50, truncated: false });
  assert.equal((await dirSize(tmp, 3)).truncated, true);
  fs.rmSync(tmp, { recursive: true, force: true });
});
