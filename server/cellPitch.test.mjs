import test from 'node:test';
import assert from 'node:assert/strict';
import { estimateCellAxisPitchCandidates } from './cellPitch.ts';

test('cell pitch candidates exclude fine 6px busbar periodicity while retaining whole-cell seam pitch', () => {
  const n = 720;
  const profile = new Float32Array(n);
  for (let x = 0; x < n; x++) {
    profile[x] = 0.8 * (x % 6 === 0 ? 1 : 0) + 1.2 * (x % 35 === 0 ? 1 : 0);
  }
  const candidates = estimateCellAxisPitchCandidates(profile);
  assert.ok(candidates.every((candidate) => candidate.pitch >= 14));
  assert.ok(candidates.some((candidate) => candidate.pitch >= 32 && candidate.pitch <= 38));
});
