import test from 'node:test';
import assert from 'node:assert/strict';
import { crackMarkerKind } from '../src/utils/crackMarker.ts';
import { toProbabilities } from './cellClassifier.ts';
import { shadowFlagsCell } from './moduleShadow.ts';
import { normalizeTrainingSettings, DEFAULT_TRAINING_SETTINGS } from './trainingSettings.ts';

test('outer-ring detections are red even when confident, low confidence is yellow, the rest magenta', () => {
  assert.equal(crackMarkerKind({ outerRing: true, confidence: 0.99 }), 'outer');
  assert.equal(crackMarkerKind({ outerRing: true, confidence: 0.4 }), 'outer');
  assert.equal(crackMarkerKind({ confidence: 0.89 }), 'low');
  assert.equal(crackMarkerKind({ confidence: 0.9 }), 'ok'); // the threshold itself is not "low"
  assert.equal(crackMarkerKind({ confidence: 0.5 }, 40), 'ok'); // exposed threshold is honoured
  assert.equal(crackMarkerKind({ source: 'user', outerRing: true, confidence: 0.1 }), 'user');
  assert.equal(crackMarkerKind({ confidence: 0.1 }, Number.NaN), 'ok'); // a broken setting must not paint everything yellow
});

test('classifier probabilities: softmax output is kept, raw logits are normalised', () => {
  const kept = toProbabilities([0.2, 0.8, 0.9, 0.1], 2, 2);
  assert.ok(Math.abs(kept[1] - 0.8) < 1e-6 && Math.abs(kept[2] - 0.9) < 1e-6);
  const soft = toProbabilities([0, 2, 5, 5], 2, 2);
  assert.ok(Math.abs(soft[0] + soft[1] - 1) < 1e-6 && soft[1] > 0.85 && Math.abs(soft[2] - 0.5) < 1e-6);
});

test('shadow detector agreement: centre inside or overlap, otherwise no', () => {
  const cell = { left: 100, top: 100, width: 50, height: 50 };
  assert.equal(shadowFlagsCell([{ left: 120, top: 120, width: 10, height: 10, confidence: 0.5 }], cell), true);
  assert.equal(shadowFlagsCell([{ left: 400, top: 400, width: 30, height: 30, confidence: 0.9 }], cell), false);
  assert.equal(shadowFlagsCell([], cell), false);
});

test('new settings: defaults, older files without them, clamping and allow-lists', () => {
  assert.equal(DEFAULT_TRAINING_SETTINGS.crackDetection.lowConfidenceThreshold, 90);
  assert.equal(DEFAULT_TRAINING_SETTINGS.crackNet.shadowEnabled, false);
  const legacy = normalizeTrainingSettings({ crackDetection: { decisionThreshold: 0.4 } }); // pre-1.4.14 file
  assert.equal(legacy.crackDetection.lowConfidenceThreshold, 90);
  assert.equal(legacy.crackNet.clsBase, 'yolo11n-cls.pt');
  const odd = normalizeTrainingSettings({ crackDetection: { lowConfidenceThreshold: 250 }, crackNet: { clsBase: 'evil.pt', targetRecall: 5, shadowEnabled: 'yes' } });
  assert.equal(odd.crackDetection.lowConfidenceThreshold, 100);
  assert.equal(odd.crackNet.clsBase, 'yolo11n-cls.pt');
  assert.equal(odd.crackNet.targetRecall, 1);
  assert.equal(odd.crackNet.shadowEnabled, false);
});
