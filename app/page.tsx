/**
 * Dashboard / home. The hero shows the tool actually working — a real sample
 * model output (the specimen) beside its rubric verdict on the calibration
 * scale — then the position-bias audit run both ways. Below: summary stats,
 * the score charts, and a data-dense results table. Everything renders from
 * baked sample data, no backend, instant on first load.
 */

import Link from 'next/link';
import {
  batchResults,
  singleScoringExamples,
  comparisonExamples,
  getDashboardStats,
  getCriterionAverages,
  getScoreDistribution,
  getDomainAverages,
} from '@/lib/sample-data';
import {
  CriterionAverageChart,
  ScoreDistributionChart,
  DomainAverageChart,
} from '@/components/ScoreChart';
import { CalibrationReadout, CriterionBar } from '@/components/CalibrationScale';
import ComparisonResult from '@/components/ComparisonResult';
import ResultsTable from '@/components/TestCaseTable';

function StatTile({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="card p-5">
      <p className="text-xs font-medium uppercase tracking-[0.12em] text-muted">{label}</p>
      <p className="mt-3 font-mono text-3xl font-semibold tracking-tight text-ink tabular-nums">{value}</p>
      <p className="mt-1 text-xs text-muted">{hint}</p>
    </div>
  );
}

export default function DashboardPage() {
  const stats = getDashboardStats();
  const criterionAverages = getCriterionAverages();
  const distribution = getScoreDistribution();
  const domainAverages = getDomainAverages();

  // The hero specimen: a clean-looking answer that is quietly wrong, so the
  // instrument visibly separates "well-presented" from "correct".
  const specimen =
    singleScoringExamples.find((e) => e.id === 'single-math-linear') ?? singleScoringExamples[0];
  // A baked comparison that demonstrates position bias for the audit panel.
  const biasCase = comparisonExamples.find((c) => c.positionBias) ?? comparisonExamples[0];

  return (
    <div className="space-y-12">
      {/* Hero — the product working: a specimen under measurement on the bench. */}
      <section className="card-elevated overflow-hidden">
        <div className="border-b border-hairline p-6 sm:p-8">
          <span className="kicker">LLM-as-judge · stateless</span>
          <h1 className="mt-3 max-w-3xl text-3xl font-semibold leading-tight tracking-tight text-ink sm:text-4xl">
            Score model outputs against a rubric — and check the judge for bias.
          </h1>
          <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-muted">
            A pre-computed run across {stats.domainCount} domains renders instantly, with no database. Then
            judge your own outputs live: rubric scoring on accuracy, clarity, and completeness, or a pairwise
            comparison that runs both orderings to catch position bias.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link href="/evaluate" className="btn-primary">
              Score a single output
              <span aria-hidden>→</span>
            </Link>
            <Link href="/compare" className="btn-secondary">
              A/B compare with bias check
            </Link>
          </div>
          <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-2 font-mono text-xs text-muted">
            <span className="inline-flex items-center gap-2">
              <span className="h-1.5 w-1.5 bg-petrol" /> no database
            </span>
            <span className="inline-flex items-center gap-2">
              <span className="h-1.5 w-1.5 bg-petrol" /> works with no API key
            </span>
            <span className="inline-flex items-center gap-2">
              <span className="h-1.5 w-1.5 bg-petrol" /> deployed on Vercel
            </span>
          </div>
        </div>

        {/* The bench: specimen on the left, calibrated verdict on the right. */}
        <div className="grid gap-px bg-hairline lg:grid-cols-[1fr_1.05fr]">
          <div className="bg-surface p-6 sm:p-8">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">Specimen</span>
              <span className="inline-flex rounded border border-hairline bg-paper px-2 py-0.5 text-xs font-medium text-muted">
                {specimen.domain}
              </span>
            </div>
            <p className="mt-4 text-xs font-semibold uppercase tracking-wide text-muted">Prompt</p>
            <p className="mt-1 text-sm leading-relaxed text-ink">{specimen.prompt}</p>
            <p className="mt-4 text-xs font-semibold uppercase tracking-wide text-muted">Model output</p>
            <p className="mt-1 whitespace-pre-wrap rounded-md border border-hairline bg-paper p-3 font-mono text-xs leading-relaxed text-ink">
              {specimen.output}
            </p>
            <p className="mt-4 text-xs text-muted">
              Judged by <span className="font-mono text-ink">{specimen.model_used}</span> — the output is scored
              as-is, never regenerated.
            </p>
          </div>

          <div className="cal-animate bg-surface p-6 sm:p-8">
            <span className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">Rubric verdict</span>
            <div className="mt-4">
              <CalibrationReadout score={specimen.total_score} />
            </div>
            <div className="mt-4 divide-y divide-hairline border-t border-hairline">
              <CriterionBar label="Accuracy" score={specimen.accuracy.score} reasoning={specimen.accuracy.reasoning} />
              <CriterionBar label="Clarity" score={specimen.clarity.score} reasoning={specimen.clarity.reasoning} />
              <CriterionBar
                label="Completeness"
                score={specimen.completeness.score}
                reasoning={specimen.completeness.reasoning}
              />
            </div>
          </div>
        </div>
      </section>

      {/* Position-bias audit — the same comparison run both ways, on the page. */}
      <section className="space-y-4">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-ink">Audit the judge for position bias</h2>
          <p className="mt-1 max-w-2xl text-sm text-muted">
            LLM judges often favour whichever output is shown first. This sample runs one comparison both ways —
            A-then-B and B-then-A — and flags the disagreement when the verdict fails to survive the swap.
          </p>
        </div>
        <ComparisonResult comparison={biasCase} />
      </section>

      {/* Summary stats */}
      <section className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatTile label="Evaluations" value={String(stats.totalEvaluations)} hint={`across ${stats.domainCount} domains`} />
        <StatTile label="Average total" value={stats.averageTotal.toFixed(2)} hint="mean score, 0–10" />
        <StatTile label="Pairwise checks" value={String(stats.comparisonCount)} hint="run in both orderings" />
        <StatTile label="Position-bias rate" value={`${stats.positionBiasRate}%`} hint="flipped on reorder" />
      </section>

      {/* Charts */}
      <section className="space-y-4">
        <div className="grid gap-4 lg:grid-cols-2">
          <CriterionAverageChart data={criterionAverages} />
          <ScoreDistributionChart data={distribution} />
        </div>
        <DomainAverageChart data={domainAverages} />
      </section>

      {/* Results table */}
      <section className="space-y-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-ink">All evaluations</h2>
          <p className="text-sm text-muted">Click any row to see the output and the judge&rsquo;s per-criterion reasoning.</p>
        </div>
        <ResultsTable rows={batchResults} />
      </section>
    </div>
  );
}
