/**
 * Shared, dependency-free helpers for presenting scores consistently across
 * every surface. Quality is encoded with a restrained semantic palette
 * (emerald / amber / rose) layered over the slate + indigo base.
 */

export interface ScoreTone {
  /** Text color class. */
  text: string;
  /** Solid background (for pills/bars). */
  bg: string;
  /** Soft tinted background + text (for chips). */
  soft: string;
  /** Hex used by Recharts bars. */
  hex: string;
}

export function scoreTone(score: number): ScoreTone {
  if (score >= 8) {
    return { text: 'text-emerald-700', bg: 'bg-emerald-600', soft: 'bg-emerald-50 text-emerald-700', hex: '#059669' };
  }
  if (score >= 6) {
    return { text: 'text-amber-700', bg: 'bg-amber-500', soft: 'bg-amber-50 text-amber-700', hex: '#d97706' };
  }
  if (score >= 4) {
    return { text: 'text-orange-700', bg: 'bg-orange-500', soft: 'bg-orange-50 text-orange-700', hex: '#ea580c' };
  }
  return { text: 'text-rose-700', bg: 'bg-rose-600', soft: 'bg-rose-50 text-rose-700', hex: '#e11d48' };
}

/** One decimal place for display; the stored value keeps full precision. */
export function formatScore(score: number): string {
  return score.toFixed(1);
}
