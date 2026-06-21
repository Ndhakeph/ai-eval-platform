/**
 * Dead-simple in-memory, per-IP rate limiter for the public live-judge
 * endpoints. This is a public demo, so the goal is to blunt casual abuse and
 * runaway cost — not to be a distributed quota system.
 *
 * Caveat (by design): serverless functions don't share memory across instances,
 * so this limits per warm instance rather than globally. That's an acceptable,
 * honest trade-off for a portfolio demo and is documented in the README.
 */

interface Bucket {
  /** Request timestamps (ms) within the current window. */
  hits: number[];
}

const buckets = new Map<string, Bucket>();

const WINDOW_MS = 60 * 60 * 1000; // 1 hour
const MAX_REQUESTS = 10; // per IP per window

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  limit: number;
  /** Epoch ms when the oldest hit ages out and a slot frees up. */
  resetAt: number;
}

/**
 * Derive a best-effort client IP from proxy headers (Vercel sets
 * `x-forwarded-for`). Falls back to a shared bucket when nothing is present.
 */
export function getClientIp(headers: Headers): string {
  const forwarded = headers.get('x-forwarded-for');
  if (forwarded) return forwarded.split(',')[0].trim();
  return headers.get('x-real-ip')?.trim() || 'unknown';
}

/**
 * Record a hit and report whether it is allowed. Call once per live request.
 */
export function checkRateLimit(
  ip: string,
  now: number = Date.now(),
  limit: number = MAX_REQUESTS,
  windowMs: number = WINDOW_MS,
): RateLimitResult {
  const bucket = buckets.get(ip) ?? { hits: [] };

  // Drop hits older than the window.
  bucket.hits = bucket.hits.filter((t) => now - t < windowMs);

  if (bucket.hits.length >= limit) {
    buckets.set(ip, bucket);
    const resetAt = bucket.hits[0] + windowMs;
    return { allowed: false, remaining: 0, limit, resetAt };
  }

  bucket.hits.push(now);
  buckets.set(ip, bucket);

  // Opportunistic cleanup so the map doesn't grow unbounded on a long-lived
  // instance — drop any wholly-expired buckets.
  if (buckets.size > 5000) {
    for (const [key, b] of buckets) {
      if (b.hits.every((t) => now - t >= windowMs)) buckets.delete(key);
    }
  }

  return {
    allowed: true,
    remaining: limit - bucket.hits.length,
    limit,
    resetAt: now + windowMs,
  };
}
