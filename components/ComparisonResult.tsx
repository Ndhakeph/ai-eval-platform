'use client';

/**
 * Renders a pairwise A/B comparison: the headline position-bias verdict, then
 * both presentation orders side by side so the (dis)agreement is visible. This
 * is the platform's differentiator — the bias audit is made explicit, not
 * buried. Winners are marked in petrol (A) / ink (B); the verdict uses the
 * score-scale rust (bias) and green (robust).
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
      ? 'bg-petrol/10 text-petrol'
      : winner === 'B'
      ? 'bg-ink text-surface'
      : 'border border-hairline text-muted';
  const label = winner === 'tie' ? 'Tie' : `Output ${winner}`;
  return <span className={`inline-flex rounded px-2 py-0.5 text-xs font-semibold ${style}`}>{label}</span>;
}

function OrderingColumn({ ordering }: { ordering: OrderingResult }) {
  const firstLabel = ordering.firstShown;
  const secondLabel = firstLabel === 'A' ? 'B' : 'A';
  return (
    <div className="card p-4">
      <div className="flex items-center justify-between border-b border-hairline pb-3">
        <h4 className="text-sm font-semibold text-ink">Order: {firstLabel} shown first</h4>
        <span className="font-mono text-xs text-muted">
          1 = {firstLabel} · 2 = {secondLabel}
        </span>
      </div>
      <div className="divide-y divide-hairline">
        {CRITERIA.map(({ key, label }) => (
          <div key={key} className="py-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-ink">{label}</span>
              <WinnerChip winner={ordering[key].winner} />
            </div>
            <p className="mt-1.5 text-xs leading-relaxed text-muted">{ordering[key].reasoning}</p>
          </div>
        ))}
        <div className="py-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-ink">Overall</span>
            <WinnerChip winner={ordering.overall.winner} />
          </div>
          <p className="mt-1.5 text-xs leading-relaxed text-muted">{ordering.overall.reasoning}</p>
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
        <div
          className="rounded-md border border-l-4 border-hairline bg-surface p-4"
          style={{ borderLeftColor: '#B14430' }}
        >
          <p className="text-sm font-semibold" style={{ color: '#B14430' }}>
            Bias detected — the verdict flips with order
          </p>
          <p className="mt-1 text-sm text-ink">
            The judge picked{' '}
            <strong>Output {orderingAB.overall.winner === 'tie' ? '—' : orderingAB.overall.winner}</strong>{' '}
            when A was shown first, but{' '}
            <strong>Output {orderingBA.overall.winner === 'tie' ? 'a tie' : orderingBA.overall.winner}</strong>{' '}
            when B was shown first. Because the verdict changed with presentation order, the result is
            unreliable — treat these outputs as effectively tied rather than trusting either single ordering.
          </p>
        </div>
      ) : (
        <div
          className="rounded-md border border-l-4 border-hairline bg-surface p-4"
          style={{ borderLeftColor: '#2F7D55' }}
        >
          <p className="text-sm font-semibold" style={{ color: '#2F7D55' }}>
            {consistentWinner === 'tie' ? 'Robust: consistent tie' : `Robust: Output ${consistentWinner} wins both ways`}
          </p>
          <p className="mt-1 text-sm text-ink">
            The judge reached the same overall verdict in both presentation orders, so the result is robust
            to position bias.
          </p>
        </div>
      )}

      {/* Both orderings */}
      <div className="grid gap-4 lg:grid-cols-2">
        <OrderingColumn ordering={orderingAB} />
        <OrderingColumn ordering={orderingBA} />
      </div>

      <p className="text-xs text-muted">
        Judged by <span className="font-mono text-ink">{comparison.model_used}</span>. Each comparison runs
        twice with the outputs swapped; a robust verdict should be identical both times.
      </p>
    </div>
  );
}
