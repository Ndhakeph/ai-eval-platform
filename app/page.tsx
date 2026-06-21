/**
 * Dashboard / home. Renders the baked batch evaluation set: a hero with a live
 * score gauge, summary stats, Recharts visualizations, and a data-dense results
 * table — entirely from sample data, no backend, instant on first load.
 */

import Link from 'next/link';
import {
  batchResults,
  getDashboardStats,
  getCriterionAverages,
  getScoreDistribution,
  getDomainAverages,
} from '@/lib/sample-data';
import {
  ScoreGauge,
  CriterionAverageChart,
  ScoreDistributionChart,
  DomainAverageChart,
} from '@/components/ScoreChart';
import ResultsTable from '@/components/TestCaseTable';

/* Minimal inline stroke icons — no icon dependency. */
const iconProps = {
  width: 18,
  height: 18,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.8,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
};

const Icons = {
  stack: (
    <svg {...iconProps}><path d="M12 3 3 8l9 5 9-5-9-5Z" /><path d="m3 13 9 5 9-5" /></svg>
  ),
  target: (
    <svg {...iconProps}><circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="3.2" /></svg>
  ),
  scale: (
    <svg {...iconProps}><path d="M12 4v16" /><path d="M5 7h14" /><path d="m5 7-2.5 5a3 3 0 0 0 5 0L5 7Z" /><path d="m19 7-2.5 5a3 3 0 0 0 5 0L19 7Z" /></svg>
  ),
  alert: (
    <svg {...iconProps}><path d="M10.3 4.3 2.5 18a1.8 1.8 0 0 0 1.6 2.7h15.8A1.8 1.8 0 0 0 21.5 18L13.7 4.3a1.9 1.9 0 0 0-3.4 0Z" /><path d="M12 9v4" /><path d="M12 17h.01" /></svg>
  ),
};

function StatCard({
  label,
  value,
  hint,
  icon,
}: {
  label: string;
  value: string;
  hint: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="card card-hover p-5">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-slate-500">{label}</p>
        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600">
          {icon}
        </span>
      </div>
      <p className="mt-3 text-3xl font-bold tracking-tight text-slate-900 tabular-nums">{value}</p>
      <p className="mt-1 text-xs text-slate-400">{hint}</p>
    </div>
  );
}

export default function DashboardPage() {
  const stats = getDashboardStats();
  const criterionAverages = getCriterionAverages();
  const distribution = getScoreDistribution();
  const domainAverages = getDomainAverages();

  return (
    <div className="space-y-10">
      {/* Hero */}
      <section className="card overflow-hidden">
        <div className="grid gap-8 p-6 sm:p-8 lg:grid-cols-[1.55fr_1fr] lg:items-center">
          <div>
            <span className="kicker">
              <span className="h-1.5 w-1.5 rounded-full bg-indigo-500" />
              LLM-as-Judge · Stateless
            </span>
            <h1 className="mt-3 text-3xl font-bold leading-tight tracking-tight text-slate-900 sm:text-4xl">
              Score and compare LLM outputs,
              <br className="hidden sm:block" /> with <span className="gradient-text">bias built in</span> to the method.
            </h1>
            <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-slate-600">
              A pre-computed evaluation run across {stats.domainCount} domains — judged on accuracy, clarity, and
              completeness. It renders instantly with no database. Then judge your own outputs live: rubric scoring,
              or pairwise comparison with explicit position-bias detection.
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
            <div className="mt-6 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-slate-400">
              <span className="inline-flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" /> No database
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" /> Works with no API key
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" /> Deployed on Vercel
              </span>
            </div>
          </div>

          <div className="flex flex-col items-center justify-center rounded-2xl bg-gradient-to-b from-slate-50 to-white p-6 ring-1 ring-slate-900/[0.05]">
            <span className="mb-1 inline-flex items-center gap-1.5 rounded-full bg-amber-50 px-2.5 py-1 text-[11px] font-semibold text-amber-700 ring-1 ring-amber-200">
              <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
              Sample data
            </span>
            <ScoreGauge value={stats.averageTotal} caption="Avg total / 10" />
            <div className="mt-3 grid w-full grid-cols-3 gap-2 text-center">
              {criterionAverages.map((c) => (
                <div key={c.criterion} className="rounded-lg bg-white/70 px-2 py-2 ring-1 ring-slate-900/[0.04]">
                  <p className="text-sm font-bold tabular-nums text-slate-900">{c.average.toFixed(1)}</p>
                  <p className="mt-0.5 text-[10px] font-medium uppercase tracking-wide text-slate-400">
                    {c.criterion.slice(0, 4)}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Stat cards */}
      <section className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="Evaluations" value={String(stats.totalEvaluations)} hint={`across ${stats.domainCount} domains`} icon={Icons.stack} />
        <StatCard label="Average total" value={stats.averageTotal.toFixed(2)} hint="mean score, 0–10" icon={Icons.target} />
        <StatCard label="Pairwise checks" value={String(stats.comparisonCount)} hint="run in both orderings" icon={Icons.scale} />
        <StatCard label="Position-bias rate" value={`${stats.positionBiasRate}%`} hint="flipped on reorder" icon={Icons.alert} />
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
          <h2 className="text-lg font-semibold tracking-tight text-slate-900">All evaluations</h2>
          <p className="text-sm text-slate-500">Click any row to see the output and the judge’s per-criterion reasoning.</p>
        </div>
        <ResultsTable rows={batchResults} />
      </section>
    </div>
  );
}
