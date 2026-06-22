/**
 * A compact, color-coded score chip. Used in tables and summaries.
 */

import { scoreTone, formatScore } from '@/lib/score-format';

interface ScorePillProps {
  score: number;
  /** Show as "/10" suffix for emphasis (e.g. headline totals). */
  outOfTen?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export default function ScorePill({ score, outOfTen = false, size = 'md' }: ScorePillProps) {
  const tone = scoreTone(score);
  const sizing =
    size === 'lg'
      ? 'px-3 py-1 text-base'
      : size === 'sm'
      ? 'px-1.5 py-0.5 text-xs'
      : 'px-2 py-0.5 text-sm';

  return (
    <span className={`inline-flex items-baseline gap-0.5 rounded font-mono font-semibold tabular-nums ${tone.soft} ${sizing}`}>
      {formatScore(score)}
      {outOfTen && <span className="text-[0.7em] font-medium opacity-70">/10</span>}
    </span>
  );
}
