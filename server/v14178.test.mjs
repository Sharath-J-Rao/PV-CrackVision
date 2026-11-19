import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeTrainingSettings, DEFAULT_TRAINING_SETTINGS } from './trainingSettings.ts';

test('v1.4.17.8: CPU gate defaults to auto with a very low cutoff and rejects bad values', () => {
  const d = DEFAULT_TRAINING_SETTINGS.crackDetection;
  assert.equal(d.cpuGate, 'auto');
  assert.equal(d.cpuGateCutoff, 0.0001);
  const s = normalizeTrainingSettings({ crackDetection: { engine: 'anomaly', cpuGate: 'bogus', cpuGateCutoff: 5 } }).crackDetection;
  assert.equal(s.cpuGate, 'auto');
  assert.ok(s.cpuGateCutoff <= 0.5);
  assert.equal(normalizeTrainingSettings({ crackDetection: { cpuGate: 'off' } }).crackDetection.cpuGate, 'off');
  assert.equal(normalizeTrainingSettings({}).crackDetection.cpuGate, 'auto');
});
