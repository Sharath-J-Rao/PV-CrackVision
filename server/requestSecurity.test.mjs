import test from 'node:test';
import assert from 'node:assert/strict';
import { validateRequestSecurity } from './requestSecurity.ts';

const base = {
  hostHeader: '127.0.0.1:43127',
  originHeader: 'http://127.0.0.1:43127',
  fetchSite: 'same-origin',
  remoteAddress: '127.0.0.1',
  port: 43127,
  remoteEnabled: false,
  remoteToken: '',
};

test('same-origin loopback application requests are accepted', () => {
  assert.deepEqual(validateRequestSecurity(base), { allowed: true });
});

test('cross-origin browser requests are rejected even when the peer is loopback', () => {
  const result = validateRequestSecurity({ ...base, originHeader: 'http://evil.example:43127', fetchSite: 'cross-site' });
  assert.equal(result.allowed, false);
  assert.ok(['cross-site', 'invalid-origin'].includes(result.reason));
});

test('DNS-rebinding Host names do not receive the loopback no-token exemption', () => {
  const result = validateRequestSecurity({ ...base, hostHeader: 'attacker.example:43127', originHeader: 'http://attacker.example:43127' });
  assert.equal(result.allowed, false);
  assert.equal(result.reason, 'remote-disabled');
});

test('remote clients require an explicit strong token', () => {
  const remote = { ...base, hostHeader: '10.1.2.3:43127', originHeader: 'http://10.1.2.3:43127', remoteAddress: '10.1.2.4', remoteEnabled: true, remoteToken: 'a'.repeat(40) };
  assert.equal(validateRequestSecurity(remote).reason, 'remote-token-required');
  assert.deepEqual(validateRequestSecurity({ ...remote, suppliedToken: 'a'.repeat(40) }), { allowed: true });
});

test('a loopback peer using an attacker-controlled Host is not trusted in remote mode', () => {
  const remote = { ...base, hostHeader: 'attacker.example:43127', originHeader: 'http://attacker.example:43127', remoteEnabled: true, remoteToken: 'b'.repeat(40), suppliedToken: 'b'.repeat(40) };
  assert.deepEqual(validateRequestSecurity(remote), { allowed: true });
  assert.equal(validateRequestSecurity({ ...remote, suppliedToken: '' }).reason, 'remote-token-required');
});

test('wrong port, missing Host and insecure/mismatched Origins fail closed', () => {
  assert.equal(validateRequestSecurity({ ...base, hostHeader: undefined }).reason, 'missing-host');
  assert.equal(validateRequestSecurity({ ...base, hostHeader: '127.0.0.1:43128' }).reason, 'wrong-port');
  assert.equal(validateRequestSecurity({ ...base, originHeader: 'https://127.0.0.1:43127' }).reason, 'invalid-origin');
});
