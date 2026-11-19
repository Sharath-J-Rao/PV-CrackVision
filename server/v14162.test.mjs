import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const src = fs.readFileSync(new URL('./cvPipeline.ts', import.meta.url), 'utf8');
test('missing classifier produces an explicit warning instead of a silent empty result', () => {
  assert.match(src, /(?:Crack anomaly detector|Archived crack classifier)[^`]*not available[^`]*did NOT run/); // v1.4.17: wording names the engine that was selected
  assert.match(src, /detectLearnedEdgeCracks\([^)]*warnings\)/);
});
test('package.json test script references only existing files', () => {
  const pkg = JSON.parse(fs.readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
  for (const f of pkg.scripts.test.match(/(?:server|scripts|electron)\/[\w.]+\.(?:mjs|cjs)/g)) {
    assert.ok(fs.existsSync(new URL(`../${f}`, import.meta.url)), `${f} missing`);
  }
});
