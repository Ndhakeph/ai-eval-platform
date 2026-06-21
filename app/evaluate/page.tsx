'use client';

/**
 * Score a single output. The user pastes a prompt, the model's output, and an
 * optional reference answer; the judge returns rubric scores plus written
 * per-criterion reasoning. The output is judged as-is — never regenerated.
 * Results are shown in-session only.
 */

import { useState } from 'react';
import Link from 'next/link';
import EvalCard from '@/components/EvalCard';
import InlineNotice from '@/components/InlineNotice';
import { singleScoringExamples } from '@/lib/sample-data';
import { ScoredEvaluation } from '@/types';

type ScoreResult = Pick<
  ScoredEvaluation,
  'accuracy' | 'clarity' | 'completeness' | 'total_score' | 'model_used'
>;

const FieldLabel = ({ children, optional }: { children: React.ReactNode; optional?: boolean }) => (
  <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-slate-700">
    {children}
    {optional && <span className="text-xs font-normal text-slate-400">optional</span>}
  </label>
);

export default function ScorePage() {
  const [prompt, setPrompt] = useState('');
  const [output, setOutput] = useState('');
  const [reference, setReference] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScoreResult | null>(null);
  const [resultTitle, setResultTitle] = useState<string>('Evaluation result');
  const [notice, setNotice] = useState<{ variant: 'warning' | 'error'; title: string; message: string; offerSample?: boolean } | null>(null);

  const [exampleIndex, setExampleIndex] = useState(0);

  const loadExample = () => {
    const ex = singleScoringExamples[exampleIndex % singleScoringExamples.length];
    setPrompt(ex.prompt);
    setOutput(ex.output);
    setReference(ex.reference ?? '');
    setResult(null);
    setNotice(null);
    setExampleIndex((i) => i + 1);
  };

  const showBakedSample = () => {
    const ex = singleScoringExamples[0];
    setResult(ex);
    setResultTitle(`Sample · ${ex.domain}`);
    setNotice(null);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || !output.trim()) return;
    setLoading(true);
    setResult(null);
    setNotice(null);
    try {
      const res = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, output, reference: reference || undefined }),
      });
      const data = await res.json();
      if (!res.ok) {
        if (data.code === 'no_api_key') {
          setNotice({
            variant: 'warning',
            title: 'Live judging is offline',
            message: 'No API key is configured on this deployment. You can still see what a scored result looks like.',
            offerSample: true,
          });
        } else if (data.code === 'rate_limited') {
          setNotice({
            variant: 'warning',
            title: 'Hourly limit reached',
            message: 'You’ve hit the live-evaluation limit for now. Try again later, or view a sample result.',
            offerSample: true,
          });
        } else {
          setNotice({ variant: 'error', title: 'Could not score this output', message: data.error ?? 'Please try again in a moment.' });
        }
        return;
      }
      setResult(data.result as ScoreResult);
      setResultTitle('Evaluation result');
    } catch {
      setNotice({ variant: 'error', title: 'Network error', message: 'Could not reach the judge. Check your connection and try again.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">Score a single output</h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-600">
          The judge scores the output you provide on accuracy, clarity, and completeness (0-10 each) with written
          reasoning per criterion. It evaluates exactly what you paste — it never rewrites or regenerates the output.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Form */}
        <form onSubmit={submit} className="space-y-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-900">Input</h2>
            <button type="button" onClick={loadExample} className="text-sm font-medium text-indigo-600 hover:text-indigo-700">
              Load an example
            </button>
          </div>

          <div>
            <FieldLabel>Prompt</FieldLabel>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              placeholder="The task or question that was given to the model…"
              className="w-full resize-y rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900 shadow-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
            />
          </div>

          <div>
            <FieldLabel>Model output</FieldLabel>
            <textarea
              value={output}
              onChange={(e) => setOutput(e.target.value)}
              rows={5}
              placeholder="The output you want judged…"
              className="w-full resize-y rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900 shadow-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
            />
          </div>

          <div>
            <FieldLabel optional>Reference answer</FieldLabel>
            <textarea
              value={reference}
              onChange={(e) => setReference(e.target.value)}
              rows={3}
              placeholder="A gold answer to judge against. Leave blank to judge on merits alone."
              className="w-full resize-y rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900 shadow-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
            />
          </div>

          <button
            type="submit"
            disabled={loading || !prompt.trim() || !output.trim()}
            className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {loading ? 'Judging…' : 'Score output'}
          </button>
        </form>

        {/* Result */}
        <div className="space-y-4">
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
                    Show a sample scored result →
                  </button>
                ) : undefined
              }
            />
          )}

          {loading && (
            <div className="space-y-3 rounded-xl border border-slate-200 bg-white p-5">
              <div className="skeleton h-8 w-1/3 rounded" />
              <div className="skeleton h-3 w-full rounded" />
              <div className="skeleton h-3 w-5/6 rounded" />
              <div className="skeleton h-3 w-full rounded" />
              <div className="skeleton h-3 w-2/3 rounded" />
            </div>
          )}

          {!loading && result && <EvalCard result={result} title={resultTitle} />}

          {!loading && !result && !notice && (
            <div className="flex h-full min-h-48 items-center justify-center rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center">
              <p className="text-sm text-slate-400">
                Scores and per-criterion reasoning will appear here.{' '}
                <Link href="/" className="font-medium text-indigo-600 hover:underline">
                  See sample results
                </Link>
                .
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
