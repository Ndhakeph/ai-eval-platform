'use client';

/**
 * Renders a pairwise A/B comparison: the headline position-bias verdict, then
 * both presentation orders side by side so the (dis)agreement is visible. This
 * is the platform's differentiator — the bias handling is made explicit, not
 * buried.
 */

import { ABComparison, OrderingResult, Criterion } from '@/types';

type ComparisonView = Pick<
  ABComparison,
  'orderingAB' | 'orderingBA' | 'positionBias' | 'consistentWinner' | 'model_used'
> & { prompt?: string };

const CRITERIA: { key: Criterion; label: string }[] = [
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'clarity', label: 'Clarity' },
  { key: 'completeness', label: 'Completeness' },
];

function WinnerChip({ winner }: { winner: 'A' | 'B' | 'tie' }) {
  const style =
    winner === 'A'
      ? 'bg-indigo-100 text-indigo-700'
      : winner === 'B'
      ? 'bg-slate-800 text-white'
      : 'bg-amber-100 text-amber-700';
  const label = winner === 'tie' ? 'Tie' : `Output ${winner}`;
  return <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-semibold ${style}`}>{label}</span>;
}

function OrderingColumn({ ordering }: { ordering: OrderingResult }) {
  const firstLabel = ordering.firstShown;
  const secondLabel = firstLabel === 'A' ? 'B' : 'A';
  return (
    <div className="card p-4">
      <div className="flex items-center justify-between border-b border-slate-100 pb-3">
        <h4 className="text-sm font-semibold text-slate-900">
          Order: {firstLabel} shown first
        </h4>
        <span className="text-xs text-slate-400">
          1 = {firstLabel} · 2 = {secondLabel}
        </span>
      </div>
      <div className="divide-y divide-slate-100">
        {CRITERIA.map(({ key, label }) => (
          <div key={key} className="py-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-slate-700">{label}</span>
              <WinnerChip winner={ordering[key].winner} />
            </div>
            <p className="mt-1.5 text-xs leading-relaxed text-slate-600">{ordering[key].reasoning}</p>
          </div>
        ))}
        <div className="py-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-slate-900">Overall</span>
            <WinnerChip winner={ordering.overall.winner} />
          </div>
          <p className="mt-1.5 text-xs leading-relaxed text-slate-600">{ordering.overall.reasoning}</p>
        </div>
      </div>
    </div>
  );
}

export default function ComparisonResult({ comparison }: { comparison: ComparisonView }) {
  const { positionBias, consistentWinner, orderingAB, orderingBA } = comparison;

  return (
    <div className="animate-fade-in-up space-y-4">
      {/* Verdict banner */}
      {positionBias ? (
        <div className="rounded-2xl border border-amber-300 bg-amber-50 p-4 shadow-[0_10px_30px_-18px_rgba(217,119,6,0.5)]">
          <div className="flex items-start gap-3">
            <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-amber-400 text-sm font-bold text-white">
              !
            </span>
            <div>
              <p className="text-sm font-semibold text-amber-900">Position bias detected</p>
              <p className="mt-1 text-sm text-amber-800">
                The judge picked <strong>Output {orderingAB.overall.winner === 'tie' ? '—' : orderingAB.overall.winner}</strong>{' '}
                when A was shown first, but{' '}
                <strong>Output {orderingBA.overall.winner === 'tie' ? 'a tie' : orderingBA.overall.winner}</strong>{' '}
                when B was shown first. Because the verdict changed with presentation order, the result is unreliable —
                treat these outputs as effectively tied rather than trusting either single ordering.
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="rounded-2xl border border-emerald-300 bg-emerald-50 p-4 shadow-[0_10px_30px_-18px_rgba(5,150,105,0.5)]">
          <div className="flex items-start gap-3">
            <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-emerald-500 text-sm font-bold text-white">
              ✓
            </span>
            <div>
              <p className="text-sm font-semibold text-emerald-900">
                {consistentWinner === 'tie' ? 'Consistent tie' : `Consistent winner: Output ${consistentWinner}`}
              </p>
              <p className="mt-1 text-sm text-emerald-800">
                The judge reached the same overall verdict in both presentation orders, so the result is robust to
                position bias.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Both orderings */}
      <div className="grid gap-4 lg:grid-cols-2">
        <OrderingColumn ordering={orderingAB} />
        <OrderingColumn ordering={orderingBA} />
      </div>

      <p className="text-xs text-slate-400">
        Judged by <span className="font-mono text-slate-500">{comparison.model_used}</span>. Each comparison runs twice
        with the outputs swapped; a robust verdict should be identical both times.
      </p>
    </div>
  );
}
