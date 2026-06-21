/**
 * Dashboard / home. Renders the baked batch evaluation set: summary stats,
 * Recharts visualizations, and a data-dense results table — entirely from
 * sample data, no backend, instant on first load.
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
  CriterionAverageChart,
  ScoreDistributionChart,
  DomainAverageChart,
} from '@/components/ScoreChart';
import ResultsTable from '@/components/TestCaseTable';

function StatCard({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-sm font-medium text-slate-500">{label}</p>
      <p className="mt-2 text-3xl font-bold tracking-tight text-slate-900 tabular-nums">{value}</p>
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
      {/* Intro */}
      <section>
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Evaluation dashboard</h1>
          <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-50 px-2.5 py-1 text-xs font-semibold text-amber-700 ring-1 ring-amber-200">
            <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
            Sample data
          </span>
        </div>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-600">
          A pre-computed evaluation run across {stats.domainCount} domains, judged on accuracy, clarity, and
          completeness. This renders instantly with no database — then try the live tools to judge your own outputs.
        </p>
        <div className="mt-4 flex flex-wrap gap-3">
          <Link
            href="/evaluate"
            className="inline-flex rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-indigo-700"
          >
            Score a single output →
          </Link>
          <Link
            href="/compare"
            className="inline-flex rounded-md bg-white px-4 py-2 text-sm font-semibold text-slate-700 ring-1 ring-slate-300 transition-colors hover:bg-slate-50"
          >
            A/B compare with bias check →
          </Link>
        </div>
      </section>

      {/* Stat cards */}
      <section className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard
          label="Evaluations"
          value={String(stats.totalEvaluations)}
          hint={`across ${stats.domainCount} domains`}
        />
        <StatCard
          label="Average total"
          value={stats.averageTotal.toFixed(2)}
          hint="mean score, 0-10"
        />
        <StatCard
          label="Pairwise checks"
          value={String(stats.comparisonCount)}
          hint="run in both orderings"
        />
        <StatCard
          label="Position-bias rate"
          value={`${stats.positionBiasRate}%`}
          hint="comparisons that flipped on reorder"
        />
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
          <h2 className="text-lg font-semibold text-slate-900">All evaluations</h2>
          <p className="text-sm text-slate-500">Click any row to see the output and the judge’s per-criterion reasoning.</p>
        </div>
        <ResultsTable rows={batchResults} />
      </section>
    </div>
  );
}
