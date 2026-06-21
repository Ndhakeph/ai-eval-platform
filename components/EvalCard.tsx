'use client';

/**
 * Detailed result card for a single scored output: the headline total plus each
 * criterion's score, a proportional bar, and the judge's written reasoning.
 * Shared by the Score Output page (live results) and the dashboard showcase.
 */

import { ScoredEvaluation } from '@/types';
import { scoreTone, formatScore } from '@/lib/score-format';

type ScoreLike = Pick<
  ScoredEvaluation,
  'accuracy' | 'clarity' | 'completeness' | 'total_score' | 'model_used'
>;

function CriterionRow({
  label,
  score,
  reasoning,
}: {
  label: string;
  score: number;
  reasoning: string;
}) {
  const tone = scoreTone(score);
  return (
    <div className="py-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-slate-700">{label}</span>
        <span className={`text-sm font-semibold tabular-nums ${tone.text}`}>
          {formatScore(score)}<span className="text-xs text-slate-400">/10</span>
        </span>
      </div>
      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full rounded-full ${tone.bg}`} style={{ width: `${(score / 10) * 100}%` }} />
      </div>
      <p className="mt-2 text-sm leading-relaxed text-slate-600">{reasoning}</p>
    </div>
  );
}

export default function EvalCard({
  result,
  title,
}: {
  result: ScoreLike;
  title?: string;
}) {
  const tone = scoreTone(result.total_score);

  return (
    <div className="animate-fade-in-up rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
            {title ?? 'Evaluation result'}
          </p>
          <p className="mt-1 text-sm text-slate-500">
            Judged by <span className="font-mono text-slate-700">{result.model_used}</span>
          </p>
        </div>
        <div className="text-right">
          <div className={`text-3xl font-bold tabular-nums ${tone.text}`}>
            {formatScore(result.total_score)}
          </div>
          <div className="text-xs font-medium uppercase tracking-wide text-slate-400">Total / 10</div>
        </div>
      </div>

      <div className="divide-y divide-slate-100">
        <CriterionRow label="Accuracy" score={result.accuracy.score} reasoning={result.accuracy.reasoning} />
        <CriterionRow label="Clarity" score={result.clarity.score} reasoning={result.clarity.reasoning} />
        <CriterionRow
          label="Completeness"
          score={result.completeness.score}
          reasoning={result.completeness.reasoning}
        />
      </div>
    </div>
  );
}
