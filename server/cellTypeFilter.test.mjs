import test from 'node:test';
import assert from 'node:assert/strict';
import { filterCellsByType, learnProfileFromBoxes } from './cellTypeFilter.ts';

const base = { enabled: true, relativeMin: 0.5, relativeMax: 1.8, profiles: [
  { id: 'a', name: 'std', mode: 'accept', enabled: true, aspect: 1, aspectTolerance: 0.3, minLongPx: 0, maxLongPx: 0 },
] };
const sq = (n = 40) => ({ left: 0, top: 0, width: n, height: n });

test('accepts square cells, rejects elongated and oversized boxes', () => {
  const boxes = [sq(), sq(41), sq(39), { left: 0, top: 0, width: 80, height: 40 }, sq(200)];
  const { kept, rejected } = filterCellsByType(boxes, base);
  assert.equal(kept.length, 3);
  assert.equal(rejected.length, 2);
});
test('extra accept type (half-cut) lets 2:1 cells through', () => {
  const s = { ...base, profiles: [...base.profiles, { id: 'h', name: 'half', mode: 'accept', enabled: true, aspect: 2, aspectTolerance: 0.25, minLongPx: 0, maxLongPx: 0 }] };
  // half-cut keeps the full cell's long side (40) and halves the short side (20)
  const { kept } = filterCellsByType([sq(), sq(), sq(), { left: 0, top: 0, width: 40, height: 20 }], s);
  assert.equal(kept.length, 4);
});
test('reject profile drops matching boxes; disabled filter passes everything', () => {
  const s = { ...base, profiles: [{ id: 'r', name: 'junk', mode: 'reject', enabled: true, aspect: 1, aspectTolerance: 0.3, minLongPx: 100, maxLongPx: 0 }] };
  assert.equal(filterCellsByType([sq(), sq(), sq(), sq(150)], s).kept.length, 3);
  assert.equal(filterCellsByType([sq(), sq(300)], { ...base, enabled: false }).kept.length, 2);
});
test('learn profile needs >=4 boxes and measures aspect', () => {
  assert.equal(learnProfileFromBoxes([sq(), sq()]), null);
  const l = learnProfileFromBoxes([1, 2, 3, 4].map(() => ({ left: 0, top: 0, width: 60, height: 40 })));
  assert.equal(l.aspect, 1.5);
});

// ---- v1.4.8.7: undersized / low-confidence detections are rejected ----
import { DEFAULT_TRAINING_SETTINGS as __DEFAULTS } from './trainingSettings.ts';
test('v1.4.8.7 undersize gates reject fragments, tiny boxes and low-confidence detections', async () => {
  const { filterCellsByType } = await import('./cellTypeFilter.ts');
  const settings = structuredClone(__DEFAULTS.cellTypeFilter);
  const cell = (i, conf = 0.9, size = 100) => ({ left: i * 110, top: 0, width: size, height: size, confidence: conf });
  const boxes = [...Array.from({ length: 8 }, (_, i) => cell(i)),
    { left: 0, top: 200, width: 100, height: 52, confidence: 0.9 },   // half-height fragment (area 0.52x) -> rejected by area gate
    { left: 200, top: 200, width: 9, height: 9, confidence: 0.9 },    // below absolute minimum short side
    { left: 400, top: 200, width: 100, height: 100, confidence: 0.01 }, // full-size but below confidence floor
  ];
  const { kept, rejected } = filterCellsByType(boxes, settings);
  assert.equal(kept.length, 8);
  assert.equal(rejected.length, 3);
  const off = filterCellsByType(boxes, { ...settings, minShortPx: 0, minAreaRelative: 0, minShortRelative: 0, minConfidence: 0 });
  assert.equal(off.kept.length, 9, 'with the new gates off only the legacy stages apply (low-confidence box now passes)');
});
test('v1.4.8.7 fragments do not drag the reference cell down (confident half defines it)', async () => {
  const { filterCellsByType } = await import('./cellTypeFilter.ts');
  const settings = structuredClone(__DEFAULTS.cellTypeFilter);
  const full = Array.from({ length: 5 }, (_, i) => ({ left: i * 110, top: 0, width: 100, height: 100, confidence: 0.9 }));
  const frags = Array.from({ length: 5 }, (_, i) => ({ left: i * 110, top: 300, width: 100, height: 45, confidence: 0.2 }));
  const { kept } = filterCellsByType([...full, ...frags], settings);
  assert.equal(kept.length, 5);
});
