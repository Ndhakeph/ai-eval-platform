/**
 * Low-level LLM client for the judge, talking to FastRouter's
 * OpenAI-compatible API.
 *
 * This module is deliberately thin: it owns the client, reports whether a real
 * key is configured, and exposes one helper that requests JSON and returns the
 * raw string. All scoring/parsing logic lives in `lib/judge.ts`.
 *
 * The app must build and boot with NO env vars set, so we never throw at module
 * load — a missing key just means `isJudgeConfigured()` returns false and the
 * live endpoints degrade to a friendly "add an API key" state.
 */

import OpenAI from 'openai';

const RAW_KEY = process.env.FASTROUTER_API_KEY?.trim();

/** Model used for judging, configurable via env. */
export const MODEL_NAME = process.env.LLM_MODEL?.trim() || 'anthropic/claude-sonnet-4.6';

const PLACEHOLDERS = new Set([
  '',
  'placeholder',
  'placeholder-api-key',
  'your_fastrouter_api_key_here',
]);

/** True only when a real (non-placeholder) API key is present. */
export function isJudgeConfigured(): boolean {
  return !!RAW_KEY && !PLACEHOLDERS.has(RAW_KEY);
}

// Instantiate with a harmless placeholder when unconfigured so importing this
// module never throws. Calls are gated on isJudgeConfigured() before use.
const client = new OpenAI({
  apiKey: RAW_KEY || 'placeholder-api-key',
  baseURL: 'https://api.fastrouter.ai/api/v1',
});

interface JudgeChatOptions {
  system: string;
  user: string;
  /** Lower is more deterministic; judging wants 0. */
  temperature?: number;
}

/**
 * Send a judging prompt and return the raw assistant text.
 *
 * We first ask for strict JSON mode (`response_format: json_object`). Some
 * models proxied through FastRouter reject that parameter, so on failure we
 * transparently retry once without it and lean on the prompt + defensive
 * parsing in `lib/judge.ts` instead. Either way the caller gets a string.
 */
export async function judgeChat({ system, user, temperature = 0 }: JudgeChatOptions): Promise<string> {
  const messages = [
    { role: 'system' as const, content: system },
    { role: 'user' as const, content: user },
  ];

  try {
    const res = await client.chat.completions.create({
      model: MODEL_NAME,
      messages,
      temperature,
      response_format: { type: 'json_object' },
    });
    return res.choices[0]?.message?.content ?? '';
  } catch {
    // Retry without JSON mode for models that don't support response_format.
    const res = await client.chat.completions.create({
      model: MODEL_NAME,
      messages,
      temperature,
    });
    return res.choices[0]?.message?.content ?? '';
  }
}
