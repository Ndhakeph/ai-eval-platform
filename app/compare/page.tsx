'use client';

/**
 * A/B Compare — the differentiator. The judge picks a winner per criterion for
 * two outputs, run in BOTH presentation orders. When swapping the order flips
 * the verdict, that's position bias, and we surface it prominently. Results are
 * shown in-session only.
 */

import { useState } from 'react';
import Link from 'next/link';
import ComparisonResult from '@/components/ComparisonResult';
import InlineNotice from '@/components/InlineNotice';
import { comparisonExamples } from '@/lib/sample-data';
import { ABComparison } from '@/types';

type ComparisonView = Pick<
  ABComparison,
  'orderingAB' | 'orderingBA' | 'positionBias' | 'consistentWinner' | 'model_used'
> & { prompt?: string };

export default function ComparePage() {
  const [prompt, setPrompt] = useState('');
  const [outputA, setOutputA] = useState('');
  const [outputB, setOutputB] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ComparisonView | null>(null);
  const [exampleIndex, setExampleIndex] = useState(0);
  const [notice, setNotice] = useState<{ variant: 'warning' | 'error'; title: string; message: string; offerSample?: boolean } | null>(null);

  const loadExample = () => {
    const ex = comparisonExamples[exampleIndex % comparisonExamples.length];
    setPrompt(ex.prompt);
    setOutputA(ex.outputA);
    setOutputB(ex.outputB);
    setResult(null);
    setNotice(null);
    setExampleIndex((i) => i + 1);
  };

  const showBakedSample = () => {
    // Prefer a baked example that demonstrates position bias.
    const ex = comparisonExamples.find((c) => c.positionBias) ?? comparisonExamples[0];
    setResult(ex);
    setNotice(null);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || !outputA.trim() || !outputB.trim()) return;
    setLoading(true);
    setResult(null);
    setNotice(null);
    try {
      const res = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, outputA, outputB }),
      });
      const data = await res.json();
      if (!res.ok) {
        if (data.code === 'no_api_key') {
          setNotice({
            variant: 'warning',
            title: 'Live judging is offline',
            message: 'No API key is configured on this deployment. You can still explore a baked comparison that demonstrates position bias.',
            offerSample: true,
          });
        } else if (data.code === 'rate_limited') {
          setNotice({
            variant: 'warning',
            title: 'Hourly limit reached',
            message: 'You’ve hit the live-comparison limit for now. Try again later, or view a sample comparison.',
            offerSample: true,
          });
        } else {
          setNotice({ variant: 'error', title: 'Could not run this comparison', message: data.error ?? 'Please try again in a moment.' });
        }
        return;
      }
      setResult(data as ComparisonView);
    } catch {
      setNotice({ variant: 'error', title: 'Network error', message: 'Could not reach the judge. Check your connection and try again.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <span className="kicker">
          <span className="h-1.5 w-1.5 bg-petrol" />
          Pairwise · bias-aware
        </span>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink sm:text-3xl">A/B compare with position-bias check</h1>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted">
          Scores the pair in <strong>both orders</strong> — A-then-B and B-then-A — and flags any verdict that
          doesn&rsquo;t survive the swap. A result you can trust is identical both ways.
        </p>
      </div>

      {/* Input form */}
      <form onSubmit={submit} className="card space-y-4 p-5">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-ink">Input</h2>
          <button type="button" onClick={loadExample} className="text-sm font-medium text-petrol hover:text-petrol-pressed">
            Load an example
          </button>
        </div>

        <div>
          <label className="mb-1.5 block text-sm font-medium text-ink">Prompt</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={2}
            placeholder="The task both outputs are responding to…"
            className="input-field"
          />
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-ink">
              <span className="inline-flex rounded bg-petrol/10 px-2 py-0.5 text-xs font-semibold text-petrol">Output A</span>
            </label>
            <textarea
              value={outputA}
              onChange={(e) => setOutputA(e.target.value)}
              rows={6}
              placeholder="First candidate output…"
              className="input-field"
            />
          </div>
          <div>
            <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-ink">
              <span className="inline-flex rounded bg-ink px-2 py-0.5 text-xs font-semibold text-surface">Output B</span>
            </label>
            <textarea
              value={outputB}
              onChange={(e) => setOutputB(e.target.value)}
              rows={6}
              placeholder="Second candidate output…"
              className="input-field"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={loading || !prompt.trim() || !outputA.trim() || !outputB.trim()}
          className="btn-primary w-full sm:w-auto"
        >
          {loading ? 'Judging both orders…' : 'Compare (runs both orderings)'}
        </button>
      </form>

      {/* Output */}
      {notice && (
        <InlineNotice
          variant={notice.variant}
          title={notice.title}
          message={notice.message}
          action={
            notice.offerSample ? (
              <button
                onClick={showBakedSample}
                className="inline-flex rounded-md border border-hairline bg-surface px-3 py-1.5 text-sm font-medium text-ink hover:border-petrol hover:text-petrol"
              >
                Show a sample comparison →
              </button>
            ) : undefined
          }
        />
      )}

      {loading && (
        <div className="grid gap-4 lg:grid-cols-2">
          {[0, 1].map((i) => (
            <div key={i} className="card space-y-3 p-4">
              <div className="skeleton h-5 w-2/3 rounded" />
              <div className="skeleton h-3 w-full rounded" />
              <div className="skeleton h-3 w-5/6 rounded" />
              <div className="skeleton h-3 w-full rounded" />
              <div className="skeleton h-3 w-3/4 rounded" />
            </div>
          ))}
        </div>
      )}

      {!loading && result && <ComparisonResult comparison={result} />}

      {!loading && !result && !notice && (
        <div className="flex min-h-40 items-center justify-center rounded-md border border-dashed border-hairline bg-surface p-8 text-center">
          <p className="text-sm text-muted">
            The verdict for both orderings — and any position-bias flag — will appear here.{' '}
            <Link href="/" className="font-medium text-petrol hover:underline">
              See sample data
            </Link>
            .
          </p>
        </div>
      )}
    </div>
  );
}
