// Regression guards for commercial-release hygiene: privacy (no third-party requests), legal metadata, support files, junk.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildNotices } from '../scripts/generate-third-party-notices.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = (file) => fs.readFileSync(path.join(root, file), 'utf8');

test('the app makes no third-party network requests (offline + privacy): no remote fonts in HTML, CSS or CSP', () => {
  for (const file of ['index.html', 'src/index.css', 'server.ts']) {
    assert.doesNotMatch(read(file), /fonts\.googleapis\.com|fonts\.gstatic\.com/, `${file} references a remote font host`);
  }
});

test('package.json carries commercial metadata', () => {
  const pkg = JSON.parse(read('package.json'));
  assert.ok(pkg.license, 'license field is required');
  assert.ok(pkg.description, 'description is required');
});

test('THIRD_PARTY_NOTICES.md is current with package-lock.json', () => {
  const lock = JSON.parse(read('package-lock.json'));
  assert.equal(read('THIRD_PARTY_NOTICES.md'), buildNotices(lock), 'run: npm run notices');
});

test('support and security documents ship with the release', () => {
  for (const file of ['SUPPORT.md', 'SECURITY.md', 'THIRD_PARTY_NOTICES.md']) assert.ok(fs.existsSync(path.join(root, file)), `${file} is missing`);
});

test('no stray temp files in the release root', () => {
  assert.ok(!fs.existsSync(path.join(root, 'None.tmp')));
  assert.deepEqual(fs.readdirSync(root).filter((name) => name.endsWith('.tmp')), []);
});

test('lifecycle hardening is present: friendly port error, graceful shutdown, desktop crash logging', () => {
  const server = read('server.ts');
  assert.match(server, /EADDRINUSE/);
  assert.match(server, /SIGTERM/);
  const electron = read('electron/main.cjs');
  assert.match(electron, /render-process-gone/);
  assert.match(electron, /uncaughtException/);
});

test('index.html is well-formed: no unclosed <noscript> (it would swallow the module script and leave a blank page)', () => {
  const html = read('index.html');
  const opens = (html.match(/<noscript\b/gi) || []).length, closes = (html.match(/<\/noscript>/gi) || []).length;
  assert.equal(opens, closes, 'unbalanced <noscript> tags');
  assert.match(html, /<div id="root"><\/div>\s*<script type="module" src="\/src\/main\.tsx"><\/script>/);
});
