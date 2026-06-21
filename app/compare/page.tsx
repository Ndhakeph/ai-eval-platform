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
          <span className="h-1.5 w-1.5 rounded-full bg-indigo-500" />
          Pairwise · bias-aware
        </span>
        <h1 className="mt-2 text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">A/B compare with position-bias check</h1>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-600">
          LLM judges have a known failure mode: they often favor whichever output is shown first, regardless of quality.
          This tool runs the comparison <strong>both ways</strong> — A-then-B and B-then-A — and flags any disagreement.
          A verdict you can trust should survive the swap.
        </p>
      </div>

      {/* Input form */}
      <form onSubmit={submit} className="card space-y-4 p-5">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-900">Input</h2>
          <button type="button" onClick={loadExample} className="text-sm font-medium text-indigo-600 hover:text-indigo-700">
            Load an example
          </button>
        </div>

        <div>
          <label className="mb-1.5 block text-sm font-medium text-slate-700">Prompt</label>
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
            <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-slate-700">
              <span className="inline-flex rounded-full bg-indigo-100 px-2 py-0.5 text-xs font-semibold text-indigo-700">Output A</span>
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
            <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-slate-700">
              <span className="inline-flex rounded-full bg-slate-800 px-2 py-0.5 text-xs font-semibold text-white">Output B</span>
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
                className="inline-flex rounded-md bg-white px-3 py-1.5 text-sm font-medium text-slate-700 ring-1 ring-slate-300 hover:bg-slate-50"
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
        <div className="flex min-h-40 items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-white/60 p-8 text-center">
          <p className="text-sm text-slate-400">
            The verdict for both orderings — and any position-bias flag — will appear here.{' '}
            <Link href="/" className="font-medium text-indigo-600 hover:underline">
              See sample data
            </Link>
            .
          </p>
        </div>
      )}
    </div>
  );
}
