import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { originalImageName, originalImagePath } from '../src/utils/markedImage.ts';

test('v1.4.15.0 downloads keep the uploaded file name (case, spaces and dashes untouched)', () => {
  assert.equal(originalImageName('Cracked modules Training data/021 Damaged - 7cyD85jZDGi - wide - 01.JPG'), '021 Damaged - 7cyD85jZDGi - wide - 01.JPG');
  assert.equal(originalImageName('IMG_0001.jpeg'), 'IMG_0001.jpeg');
  assert.equal(originalImageName('plain.JPG'), 'plain.JPG');
  assert.equal(originalImageName('scan.png'), 'scan.jpg', 'marked output is JPEG, so only a non-JPEG extension changes');
});

test('v1.4.15.0 ZIP entries keep the uploaded folder structure', () => {
  assert.equal(originalImagePath('A folder/Sub/x - 01.JPG'), 'A folder/Sub/x - 01.JPG');
  assert.equal(originalImagePath('../../evil/x.JPG'), 'evil/x.JPG');
  assert.equal(originalImagePath('x.JPG'), 'x.JPG');
});

test('v1.4.15.0 end-user tabs: annotation / training / data tabs are advanced-only, diagnostics and marked images are not', () => {
  const app = fs.readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8');
  for (const label of ['1 · Cell Annotation', '2 · Crack Annotation', '>Training</button>', '>Training Settings</button>', 'Data &amp; Models']) {
    const at = app.indexOf(label); assert.ok(at > 0, label);
    assert.ok(app.lastIndexOf('{advancedMode && <button', at) > app.lastIndexOf('</nav>', at) || app.lastIndexOf('{advancedMode && <button', at) > at - 900, `${label} must be gated by advancedMode`);
  }
  assert.ok(!/\{advancedMode && <button\s+type="button"\s+onClick=\{\(\) => setActiveTab\('tiles'\)\}/.test(app), 'Detection Diagnostics is visible by default');
});

test('v1.4.15.0 crack detection unlocks when the cell stage settles, and failed images no longer block it', () => {
  const app = fs.readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8');
  assert.match(app, /cellDoneCount \+ cellErrorCount === frames\.length/);
  assert.match(app, /disabled=\{!cellStageSettled \|\| batchRunning\}/);
  assert.match(app, /reviewAllCellDetections/);
  assert.match(app, /autoOpenMarkedRef/);
});
