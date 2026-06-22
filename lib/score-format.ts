/**
 * Shared, dependency-free helpers for presenting scores consistently across
 * every surface. The score scale is the ONLY place colour encodes data:
 * low (0–4) rust, mid (5–7) ochre, high (8–10) green. Everything else on the
 * page is ink + paper + petrol.
 */

export interface ScoreTone {
  /** Text colour class. */
  text: string;
  /** Solid background (for bars/fills). */
  bg: string;
  /** Soft tinted background + text (for chips). */
  soft: string;
  /** Hex, for inline SVG/Recharts fills. */
  hex: string;
}

const HIGH: ScoreTone = {
  text: 'text-score-high',
  bg: 'bg-score-high',
  soft: 'bg-score-high/10 text-score-high',
  hex: '#2F7D55',
};
const MID: ScoreTone = {
  text: 'text-score-mid',
  bg: 'bg-score-mid',
  soft: 'bg-score-mid/10 text-score-mid',
  hex: '#B5862B',
};
const LOW: ScoreTone = {
  text: 'text-score-low',
  bg: 'bg-score-low',
  soft: 'bg-score-low/10 text-score-low',
  hex: '#B14430',
};

export function scoreTone(score: number): ScoreTone {
  if (score >= 8) return HIGH;
  if (score >= 5) return MID;
  return LOW;
}

/** Hex for the score band — convenience for inline styles. */
export function scoreHex(score: number): string {
  return scoreTone(score).hex;
}

/** Short band label for the calibration readout. */
export function scoreBand(score: number): 'Low' | 'Mid' | 'High' {
  if (score >= 8) return 'High';
  if (score >= 5) return 'Mid';
  return 'Low';
}

/** One decimal place for display; the stored value keeps full precision. */
export function formatScore(score: number): string {
  return score.toFixed(1);
}
