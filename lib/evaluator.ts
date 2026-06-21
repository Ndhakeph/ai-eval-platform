/**
 * Stateless batch evaluation runner.
 *
 * Scores many outputs concurrently with a bounded worker pool so realistic CSV
 * uploads finish well within Vercel's function cap. There are no artificial
 * sleeps and no persistence — results are returned to the caller and rendered
 * in-session. One failed row never sinks the batch; it is reported as an error
 * on that row instead.
 */

import { scoreOutput } from './judge';
import { ScoredEvaluation } from '@/types';

/** Max rows accepted in a single batch request (keeps us under the time cap). */
export const MAX_BATCH_ROWS = 12;

/** Concurrent judge calls in flight at once. */
const CONCURRENCY = 5;

export interface BatchItem {
  prompt: string;
  output: string;
  reference?: string;
  domain?: string;
}

export interface BatchRowResult {
  id: string;
  domain: string;
  prompt: string;
  output: string;
  reference?: string;
  /** Present on success. */
  result?: Pick<ScoredEvaluation, 'accuracy' | 'clarity' | 'completeness' | 'total_score' | 'model_used'>;
  /** Present on failure — a calm message, never a raw stack. */
  error?: string;
}

/**
 * Run a bounded-concurrency pool over `items`, scoring each one. Preserves input
 * order in the returned array.
 */
export async function runBatch(items: BatchItem[]): Promise<BatchRowResult[]> {
  const results: BatchRowResult[] = new Array(items.length);
  let cursor = 0;

  async function worker() {
    while (cursor < items.length) {
      const index = cursor++;
      const item = items[index];
      const base = {
        id: `row-${index + 1}`,
        domain: item.domain?.trim() || 'Uploaded',
        prompt: item.prompt,
        output: item.output,
        reference: item.reference,
      };

      try {
        const result = await scoreOutput({
          prompt: item.prompt,
          output: item.output,
          reference: item.reference,
        });
        results[index] = { ...base, result };
      } catch {
        results[index] = {
          ...base,
          error: 'The judge could not score this row. It was skipped so the rest of the batch could finish.',
        };
      }
    }
  }

  const pool = Array.from({ length: Math.min(CONCURRENCY, items.length) }, worker);
  await Promise.all(pool);
  return results;
}
