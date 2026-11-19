/** Same-origin and remote-token guard for the local Express application. */
import { timingSafeEqual } from 'node:crypto';
export interface RequestSecurityInput {
  hostHeader?: string;
  originHeader?: string;
  fetchSite?: string;
  remoteAddress?: string;
  port: number;
  remoteEnabled: boolean;
  remoteToken: string;
  suppliedToken?: string;
}

export interface RequestSecurityDecision {
  allowed: boolean;
  reason?: 'missing-host' | 'invalid-host' | 'wrong-port' | 'remote-disabled' | 'remote-token-required' | 'invalid-origin' | 'cross-site';
}

const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '::1']);

export function isLoopbackAddress(address: string): boolean {
  return address === '127.0.0.1' || address === '::1' || address === '::ffff:127.0.0.1';
}

function safeEqual(left: string, right: string): boolean {
  // Constant-time comparison for equal-length token byte strings.
  const a = Buffer.from(left);
  const b = Buffer.from(right);
  if (!a.length || a.length !== b.length) return false;
  return timingSafeEqual(a, b);
}


export function validateRequestSecurity(input: RequestSecurityInput): RequestSecurityDecision {
  const authority = input.hostHeader?.trim();
  if (!authority) return { allowed: false, reason: 'missing-host' };

  let host: URL;
  try { host = new URL(`http://${authority}`); }
  catch { return { allowed: false, reason: 'invalid-host' }; }
  if (host.username || host.password || host.pathname !== '/' || host.search || host.hash) {
    return { allowed: false, reason: 'invalid-host' };
  }
  if (host.port !== String(input.port)) return { allowed: false, reason: 'wrong-port' };

  const hostname = host.hostname.toLowerCase().replace(/^\[|\]$/g, '');
  const hostIsLoopback = LOOPBACK_HOSTS.has(hostname);
  const peerIsLoopback = isLoopbackAddress(input.remoteAddress ?? '');
  const localTrusted = peerIsLoopback && hostIsLoopback;

  if (!localTrusted) {
    if (!input.remoteEnabled) return { allowed: false, reason: 'remote-disabled' };
    if (!safeEqual(input.suppliedToken ?? '', input.remoteToken)) {
      return { allowed: false, reason: 'remote-token-required' };
    }
  }

  const site = input.fetchSite?.toLowerCase();
  if (site && site !== 'same-origin' && site !== 'none') return { allowed: false, reason: 'cross-site' };

  const origin = input.originHeader?.trim();
  if (origin) {
    let parsedOrigin: URL;
    try { parsedOrigin = new URL(origin); }
    catch { return { allowed: false, reason: 'invalid-origin' }; }
    if (parsedOrigin.username || parsedOrigin.password || parsedOrigin.protocol !== 'http:' || parsedOrigin.host !== host.host) {
      return { allowed: false, reason: 'invalid-origin' };
    }
  }

  return { allowed: true };
}
