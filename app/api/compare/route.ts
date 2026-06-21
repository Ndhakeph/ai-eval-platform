/**
 * Live pairwise A/B comparison endpoint.
 *
 *   POST /api/compare
 *   { prompt, outputA, outputB } -> { orderingAB, orderingBA, positionBias, consistentWinner }
 *
 * The judge runs in BOTH presentation orders so we can detect position bias —
 * when swapping which output is shown first flips the verdict. Stateless,
 * rate-limited, and degrades calmly without an API key.
 */

import { NextRequest, NextResponse } from 'next/server';
import { compareOutputs } from '@/lib/judge';
import { isJudgeConfigured } from '@/lib/llm';
import { checkRateLimit, getClientIp } from '@/lib/rate-limit';
import { ApiErrorCode } from '@/types';

export const maxDuration = 60;

function errorResponse(code: ApiErrorCode, message: string, status: number, extra?: Record<string, unknown>) {
  return NextResponse.json({ error: message, code, ...extra }, { status });
}

export async function POST(request: NextRequest) {
  if (!isJudgeConfigured()) {
    return errorResponse(
      'no_api_key',
      'Live judging is offline because no API key is configured. Explore the baked comparison instead.',
      503,
    );
  }

  const ip = getClientIp(request.headers);
  const limit = checkRateLimit(ip);
  if (!limit.allowed) {
    return errorResponse(
      'rate_limited',
      'You have reached the hourly limit for live comparisons. Please try again later, or browse the sample comparisons.',
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

  const { prompt, outputA, outputB } = (body ?? {}) as {
    prompt?: string;
    outputA?: string;
    outputB?: string;
  };

  if (!prompt?.trim() || !outputA?.trim() || !outputB?.trim()) {
    return errorResponse('bad_request', "'prompt', 'outputA', and 'outputB' are all required.", 400);
  }

  try {
    const comparison = await compareOutputs({
      prompt: prompt.trim(),
      outputA: outputA.trim(),
      outputB: outputB.trim(),
    });
    return NextResponse.json({ ...comparison, rateLimitRemaining: limit.remaining });
  } catch {
    return errorResponse(
      'judge_failed',
      'The judge could not complete this comparison. Please try again in a moment.',
      502,
    );
  }
}
