/**
 * The LLM-as-judge core.
 *
 * Two responsibilities:
 *   1. `scoreOutput` — rubric scoring of a single output (accuracy, clarity,
 *      completeness) with written per-criterion reasoning. The output is judged
 *      as given; it is never regenerated.
 *   2. `compareOutputs` — pairwise A/B judging run in BOTH presentation orders,
 *      with explicit position-bias detection.
 *
 * Everything here is stateless. JSON returned by the model is parsed
 * defensively so a stray code fence or bit of prose never fails an evaluation.
 */

import { judgeChat, MODEL_NAME } from './llm';
import {
  ScoredEvaluation,
  CriterionDetail,
  OrderingResult,
  CriterionVerdict,
} from '@/types';

/* -------------------------------------------------------------------------- */
/* Defensive JSON parsing                                                     */
/* -------------------------------------------------------------------------- */

/**
 * Extract a JSON object from arbitrary model text. Tolerates ```json fences,
 * leading/trailing prose, and smart quotes. Throws only when no object-shaped
 * substring can be found at all.
 */
export function extractJSON<T = Record<string, unknown>>(raw: string): T {
  let text = (raw ?? '').trim();

  // Strip a leading ```json / ``` fence and a trailing ``` fence if present.
  text = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '');

  // Narrow to the outermost {...} so surrounding prose is ignored.
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start !== -1 && end !== -1 && end > start) {
    text = text.slice(start, end + 1);
  }

  try {
    return JSON.parse(text) as T;
  } catch {
    // Last resort: normalize smart quotes that occasionally slip in.
    const normalized = text
      .replace(/[“”]/g, '"')
      .replace(/[‘’]/g, "'");
    return JSON.parse(normalized) as T;
  }
}

/** Coerce any model-provided value into a clean 0-10 number. */
function clampScore(value: unknown): number {
  const n = typeof value === 'number' ? value : parseFloat(String(value));
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(10, n));
}

/** Real decimal average of the three criteria, rounded to 2dp (never an int). */
export function calculateTotalScore(a: number, c: number, co: number): number {
  return Math.round(((a + c + co) / 3) * 100) / 100;
}

function asReasoning(value: unknown, fallback: string): string {
  const s = typeof value === 'string' ? value.trim() : '';
  return s || fallback;
}

function normalizeWinner(value: unknown): 'A' | 'B' | 'tie' {
  const s = String(value ?? '').trim().toLowerCase();
  if (s === 'a' || s === '1' || s === 'output a' || s === 'output 1') return 'A';
  if (s === 'b' || s === '2' || s === 'output b' || s === 'output 2') return 'B';
  return 'tie';
}

/* -------------------------------------------------------------------------- */
/* Single-output rubric scoring                                               */
/* -------------------------------------------------------------------------- */

const SCORING_SYSTEM = `You are a meticulous evaluation judge. You score a model's OUTPUT against a rubric. You never rewrite or regenerate the output — you only judge what you are given.

Score three criteria, each an integer or one-decimal number from 0 to 10:

- accuracy: Is the output factually correct and faithful to the task (and to the reference answer, if one is provided)? 0-3 mostly wrong, 4-6 partially correct, 7-8 mostly correct with minor errors, 9-10 fully correct.
- clarity: Is the output well-structured, unambiguous, and easy to follow? 0-3 confusing, 4-6 understandable but rough, 7-8 clear, 9-10 exceptionally clear.
- completeness: Does the output address everything the prompt asks for (and cover the reference's key points, if provided)? 0-3 missing most, 4-6 partial, 7-8 most points, 9-10 comprehensive.

For each criterion give one or two sentences of specific reasoning grounded in the actual text — cite what was right or wrong. Be calibrated: reserve 9-10 for genuinely excellent work and do not inflate.

Respond with ONLY a JSON object in exactly this shape, no prose, no markdown:
{
  "accuracy": {"score": <number>, "reasoning": "<text>"},
  "clarity": {"score": <number>, "reasoning": "<text>"},
  "completeness": {"score": <number>, "reasoning": "<text>"}
}`;

interface ScoreInput {
  prompt: string;
  output: string;
  reference?: string;
}

type ScoreResult = Pick<
  ScoredEvaluation,
  'accuracy' | 'clarity' | 'completeness' | 'total_score' | 'model_used'
>;

interface RawCriterion {
  score?: unknown;
  reasoning?: unknown;
}
interface RawScoreResponse {
  accuracy?: RawCriterion;
  clarity?: RawCriterion;
  completeness?: RawCriterion;
}

function detail(raw: RawCriterion | undefined): CriterionDetail {
  return {
    score: clampScore(raw?.score),
    reasoning: asReasoning(raw?.reasoning, 'No reasoning was returned for this criterion.'),
  };
}

/**
 * Score a single output against the rubric. Returns per-criterion scores +
 * reasoning and a real-decimal total. Throws only if the model returns nothing
 * parseable, in which case the API route surfaces a calm error.
 */
export async function scoreOutput({ prompt, output, reference }: ScoreInput): Promise<ScoreResult> {
  const user = [
    `# Prompt\n${prompt}`,
    reference
      ? `# Reference answer (the gold standard to compare against)\n${reference}`
      : `# Reference answer\n(none provided — judge the output on its own correctness and merits)`,
    `# Output to evaluate\n${output}`,
  ].join('\n\n');

  const raw = await judgeChat({ system: SCORING_SYSTEM, user });
  const parsed = extractJSON<RawScoreResponse>(raw);

  const accuracy = detail(parsed.accuracy);
  const clarity = detail(parsed.clarity);
  const completeness = detail(parsed.completeness);

  return {
    accuracy,
    clarity,
    completeness,
    total_score: calculateTotalScore(accuracy.score, clarity.score, completeness.score),
    model_used: MODEL_NAME,
  };
}

/* -------------------------------------------------------------------------- */
/* Pairwise A/B comparison with position-bias detection                       */
/* -------------------------------------------------------------------------- */

const COMPARE_SYSTEM = `You are a meticulous pairwise evaluation judge. Given a prompt and two candidate outputs labelled "Output 1" and "Output 2", decide which better answers the prompt.

Judge three criteria — accuracy, clarity, completeness — and an overall verdict. For each, the winner is "1", "2", or "tie", with one or two sentences of specific reasoning. Judge purely on quality; ignore which output came first and do not favour length.

Respond with ONLY a JSON object in exactly this shape, no prose, no markdown:
{
  "accuracy": {"winner": "1|2|tie", "reasoning": "<text>"},
  "clarity": {"winner": "1|2|tie", "reasoning": "<text>"},
  "completeness": {"winner": "1|2|tie", "reasoning": "<text>"},
  "overall": {"winner": "1|2|tie", "reasoning": "<text>"}
}`;

interface RawVerdict {
  winner?: unknown;
  reasoning?: unknown;
}
interface RawCompareResponse {
  accuracy?: RawVerdict;
  clarity?: RawVerdict;
  completeness?: RawVerdict;
  overall?: RawVerdict;
}

/**
 * Run the judge for a single presentation order. `first` / `second` are the
 * actual texts shown as Output 1 / Output 2; `slot1` / `slot2` map those slots
 * back to the stable A/B labels so the caller can compare across orders.
 */
async function judgePair(
  prompt: string,
  first: string,
  second: string,
  slot1: 'A' | 'B',
  slot2: 'A' | 'B',
): Promise<OrderingResult> {
  const user = [
    `# Prompt\n${prompt}`,
    `# Output 1\n${first}`,
    `# Output 2\n${second}`,
  ].join('\n\n');

  const raw = await judgeChat({ system: COMPARE_SYSTEM, user });
  const parsed = extractJSON<RawCompareResponse>(raw);

  // Map a slot-relative winner ("1"/"2"/"tie") to the stable A/B label.
  const toVerdict = (rv: RawVerdict | undefined, fallback: string): CriterionVerdict => {
    const slotWinner = normalizeWinner(rv?.winner);
    const winner = slotWinner === 'A' ? slot1 : slotWinner === 'B' ? slot2 : 'tie';
    return { winner, reasoning: asReasoning(rv?.reasoning, fallback) };
  };

  return {
    firstShown: slot1,
    accuracy: toVerdict(parsed.accuracy, 'No reasoning returned for accuracy.'),
    clarity: toVerdict(parsed.clarity, 'No reasoning returned for clarity.'),
    completeness: toVerdict(parsed.completeness, 'No reasoning returned for completeness.'),
    overall: toVerdict(parsed.overall, 'No reasoning returned for the overall verdict.'),
  };
}

interface CompareInput {
  prompt: string;
  outputA: string;
  outputB: string;
}

type CompareResult = {
  orderingAB: OrderingResult;
  orderingBA: OrderingResult;
  positionBias: boolean;
  consistentWinner: 'A' | 'B' | 'tie' | null;
  model_used: string;
};

/**
 * Compare two outputs by judging BOTH presentation orders (A-first and
 * B-first) concurrently. If the overall winner flips when the order flips, the
 * judge exhibited position bias and we flag it — the headline feature of this
 * tool.
 */
export async function compareOutputs({ prompt, outputA, outputB }: CompareInput): Promise<CompareResult> {
  const [orderingAB, orderingBA] = await Promise.all([
    // Order 1: A shown first (slot 1 = A, slot 2 = B)
    judgePair(prompt, outputA, outputB, 'A', 'B'),
    // Order 2: B shown first (slot 1 = B, slot 2 = A)
    judgePair(prompt, outputB, outputA, 'B', 'A'),
  ]);

  const winnerAB = orderingAB.overall.winner;
  const winnerBA = orderingBA.overall.winner;
  const positionBias = winnerAB !== winnerBA;

  return {
    orderingAB,
    orderingBA,
    positionBias,
    consistentWinner: positionBias ? null : winnerAB,
    model_used: MODEL_NAME,
  };
}
