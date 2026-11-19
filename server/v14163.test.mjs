import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const src = fs.readFileSync(new URL('../server.ts', import.meta.url), 'utf8');
test('dev CSP allows the Vite inline preamble while production stays strict', () => {
  assert.match(src, /script-src 'self'\$\{isProduction \? '' : " 'unsafe-inline' 'unsafe-eval'"\}/);
});
