import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const src = fs.readFileSync(new URL('../server.ts', import.meta.url), 'utf8');
test('loopback clients bypass the API rate limiter', () => {
  assert.match(src, /if \(isLoopbackAddress\(req\.socket\.remoteAddress \?\? ''\)\) return next\(\);/);
});
