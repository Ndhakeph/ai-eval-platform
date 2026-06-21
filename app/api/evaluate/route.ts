/**
 * Live single-output / batch scoring endpoint.
 *
 *   POST /api/evaluate
 *   - Single: { prompt, output, reference? }            -> { result }
 *   - Batch:  { items: [{ prompt, output, reference?, domain? }] } -> { results }
 *
 * Stateless: nothing is persisted. Guarded by a per-IP rate limit and degrades
 * to a calm, structured error when no API key is configured.
 */

import { NextRequest, NextResponse } from 'next/server';
import { scoreOutput } from '@/lib/judge';
import { runBatch, MAX_BATCH_ROWS, BatchItem } from '@/lib/evaluator';
import { isJudgeConfigured } from '@/lib/llm';
import { checkRateLimit, getClientIp } from '@/lib/rate-limit';
import { ApiErrorCode } from '@/types';

// Vercel Hobby caps serverless functions at 60s.
export const maxDuration = 60;

function errorResponse(code: ApiErrorCode, message: string, status: number, extra?: Record<string, unknown>) {
  return NextResponse.json({ error: message, code, ...extra }, { status });
}

export async function POST(request: NextRequest) {
  // 1. Degrade gracefully when no key is set — the demo still works elsewhere.
  if (!isJudgeConfigured()) {
    return errorResponse(
      'no_api_key',
      'Live judging is offline because no API key is configured. Explore the baked sample evaluation instead.',
      503,
    );
  }

  // 2. Best-effort per-IP rate limiting for this public endpoint.
  const ip = getClientIp(request.headers);
  const limit = checkRateLimit(ip);
  if (!limit.allowed) {
    return errorResponse(
      'rate_limited',
      'You have reached the hourly limit for live evaluations. Please try again later, or browse the sample results.',
      429,
      { resetAt: limit.resetAt },
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return errorResponse('bad_request', 'Request body must be valid JSON.', 400);
  }

  const { prompt, output, reference, items } = (body ?? {}) as {
    prompt?: string;
    output?: string;
    reference?: string;
    items?: BatchItem[];
  };

  try {
    // --- Batch mode ------------------------------------------------------
    if (Array.isArray(items)) {
      if (items.length === 0) {
        return errorResponse('bad_request', 'The items array is empty.', 400);
      }
      const valid = items.filter((it) => it?.prompt?.trim() && it?.output?.trim());
      if (valid.length === 0) {
        return errorResponse('bad_request', "Each item needs a non-empty 'prompt' and 'output'.", 400);
      }

      const capped = valid.slice(0, MAX_BATCH_ROWS);
      const results = await runBatch(capped);

      return NextResponse.json({
        results,
        evaluated: results.length,
        truncated: valid.length > MAX_BATCH_ROWS ? valid.length - MAX_BATCH_ROWS : 0,
        rateLimitRemaining: limit.remaining,
      });
    }

    // --- Single mode -----------------------------------------------------
    if (!prompt?.trim() || !output?.trim()) {
      return errorResponse('bad_request', "Both 'prompt' and 'output' are required.", 400);
    }

    const result = await scoreOutput({
      prompt: prompt.trim(),
      output: output.trim(),
      reference: reference?.trim() || undefined,
    });

    return NextResponse.json({ result, rateLimitRemaining: limit.remaining });
  } catch {
    return errorResponse(
      'judge_failed',
      'The judge could not complete this evaluation. Please try again in a moment.',
      502,
    );
  }
}
