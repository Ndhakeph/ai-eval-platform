'use client';

/**
 * Detailed result card for a single scored output: the headline total rendered
 * on the calibration scale, then each criterion as a calibrated bar with the
 * judge's written reasoning. Shared by the Score Output page (live results) and
 * the dashboard showcase.
 */

import { ScoredEvaluation } from '@/types';
import { CalibrationReadout, CriterionBar } from './CalibrationScale';

type ScoreLike = Pick<
  ScoredEvaluation,
  'accuracy' | 'clarity' | 'completeness' | 'total_score' | 'model_used'
>;

export default function EvalCard({
  result,
  title,
}: {
  result: ScoreLike;
  title?: string;
}) {
  return (
    <div className="card animate-fade-in-up p-5">
      <div className="flex items-start justify-between gap-4 border-b border-hairline pb-4">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-muted">
            {title ?? 'Evaluation result'}
          </p>
          <p className="mt-1 text-sm text-muted">
            Judged by <span className="font-mono text-ink">{result.model_used}</span>
          </p>
        </div>
      </div>

      <div className="border-b border-hairline py-5">
        <CalibrationReadout score={result.total_score} />
      </div>

      <div className="divide-y divide-hairline">
        <CriterionBar label="Accuracy" score={result.accuracy.score} reasoning={result.accuracy.reasoning} />
        <CriterionBar label="Clarity" score={result.clarity.score} reasoning={result.clarity.reasoning} />
        <CriterionBar
          label="Completeness"
          score={result.completeness.score}
          reasoning={result.completeness.reasoning}
        />
      </div>
    </div>
  );
}
