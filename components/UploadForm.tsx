'use client';

/**
 * Stateless batch CSV evaluator. Parses the file in the browser, sends the rows
 * to the live judge endpoint, and renders the scored results in-session. No
 * upload, no storage. Degrades calmly when the judge is offline or rate-limited.
 */

import { useRef, useState } from 'react';
import Link from 'next/link';
import { parseTestCases, validateCSVFile, generateSampleCSV } from '@/lib/csv-parser';
import { TestCaseCSVRow, ScoredEvaluation } from '@/types';
import ResultsTable from './TestCaseTable';
import InlineNotice from './InlineNotice';

const MAX_ROWS = 12; // mirrors MAX_BATCH_ROWS on the server

type Phase = 'idle' | 'parsing' | 'ready' | 'scoring' | 'done';

interface BatchRowResponse {
  id: string;
  domain: string;
  prompt: string;
  output: string;
  reference?: string;
  result?: Omit<ScoredEvaluation, 'id' | 'domain' | 'prompt' | 'output' | 'reference'>;
  error?: string;
}

export default function UploadForm() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [rows, setRows] = useState<TestCaseCSVRow[]>([]);
  const [scored, setScored] = useState<ScoredEvaluation[]>([]);
  const [skipped, setSkipped] = useState(0);
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ title: string; message: string; variant: 'warning' | 'error' } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const reset = () => {
    setRows([]);
    setScored([]);
    setSkipped(0);
    setError(null);
    setNotice(null);
    setPhase('idle');
    setFileName(null);
  };

  const handleFile = async (file: File) => {
    reset();
    const validation = validateCSVFile(file);
    if (!validation.valid) {
      setError(validation.error ?? 'Invalid file.');
      return;
    }
    setFileName(file.name);
    setPhase('parsing');
    try {
      const parsed = await parseTestCases(file);
      setRows(parsed);
      setPhase('ready');
    } catch (err) {
      setError((err as Error).message);
      setPhase('idle');
      setFileName(null);
    }
  };

  const runEvaluation = async () => {
    setPhase('scoring');
    setError(null);
    setNotice(null);
    try {
      const res = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items: rows.slice(0, MAX_ROWS) }),
      });
      const data = await res.json();

      if (!res.ok) {
        if (data.code === 'no_api_key') {
          setNotice({
            variant: 'warning',
            title: 'Live judging is offline',
            message:
              'No API key is configured on this deployment, so the batch can’t be scored live. The dashboard sample shows what scored results look like.',
          });
        } else if (data.code === 'rate_limited') {
          setNotice({
            variant: 'warning',
            title: 'Hourly limit reached',
            message: 'You’ve hit the live-evaluation limit. Try again later, or explore the sample dashboard.',
          });
        } else {
          setNotice({
            variant: 'error',
            title: 'Could not score the batch',
            message: data.error ?? 'Something went wrong. Please try again in a moment.',
          });
        }
        setPhase('ready');
        return;
      }

      const results = (data.results as BatchRowResponse[]) ?? [];
      const ok: ScoredEvaluation[] = [];
      let failed = 0;
      for (const r of results) {
        if (r.result) {
          ok.push({
            id: r.id,
            domain: r.domain,
            prompt: r.prompt,
            output: r.output,
            reference: r.reference,
            accuracy: r.result.accuracy,
            clarity: r.result.clarity,
            completeness: r.result.completeness,
            total_score: r.result.total_score,
            model_used: r.result.model_used,
          });
        } else {
          failed += 1;
        }
      }
      setScored(ok);
      setSkipped(failed);
      setPhase('done');
    } catch {
      setNotice({
        variant: 'error',
        title: 'Network error',
        message: 'Could not reach the judge. Check your connection and try again.',
      });
      setPhase('ready');
    }
  };

  const downloadSample = () => {
    const blob = new Blob([generateSampleCSV()], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample-eval-cases.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Format help */}
      <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600">
        <p>
          Upload a CSV with{' '}
          <code className="rounded bg-slate-100 px-1 py-0.5 font-mono text-xs text-slate-800">prompt</code>,{' '}
          <code className="rounded bg-slate-100 px-1 py-0.5 font-mono text-xs text-slate-800">output</code>, and an
          optional{' '}
          <code className="rounded bg-slate-100 px-1 py-0.5 font-mono text-xs text-slate-800">reference</code> column.
          Up to {MAX_ROWS} rows are scored live per run.{' '}
          <button onClick={downloadSample} className="font-medium text-indigo-600 hover:text-indigo-700 hover:underline">
            Download a sample CSV
          </button>
          .
        </p>
      </div>

      {/* Dropzone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          const f = e.dataTransfer.files[0];
          if (f) handleFile(f);
        }}
        className={`rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
          isDragging ? 'border-indigo-500 bg-indigo-50' : 'border-slate-300 bg-white hover:border-slate-400'
        }`}
      >
        <p className="text-sm text-slate-600">
          <button
            onClick={() => inputRef.current?.click()}
            className="font-semibold text-indigo-600 hover:text-indigo-700"
          >
            Choose a CSV file
          </button>{' '}
          or drag and drop
        </p>
        <p className="mt-1 text-xs text-slate-400">CSV up to 5MB{fileName ? ` · ${fileName}` : ''}</p>
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          className="sr-only"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
          }}
        />
      </div>

      {error && <InlineNotice variant="error" title="Couldn’t read that file" message={error} />}
      {notice && (
        <InlineNotice
          variant={notice.variant}
          title={notice.title}
          message={notice.message}
          action={
            <Link
              href="/"
              className="inline-flex rounded-md bg-white px-3 py-1.5 text-sm font-medium text-slate-700 ring-1 ring-slate-300 hover:bg-slate-50"
            >
              View the sample dashboard →
            </Link>
          }
        />
      )}

      {/* Parsed preview + run button */}
      {(phase === 'ready' || phase === 'scoring') && rows.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-slate-700">
              Parsed <strong>{rows.length}</strong> row{rows.length !== 1 ? 's' : ''}
              {rows.length > MAX_ROWS && (
                <span className="text-slate-500"> — the first {MAX_ROWS} will be scored</span>
              )}
              .
            </p>
            <button
              onClick={runEvaluation}
              disabled={phase === 'scoring'}
              className="inline-flex items-center gap-2 rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {phase === 'scoring' ? 'Scoring…' : `Run live evaluation`}
            </button>
          </div>
        </div>
      )}

      {/* Scoring skeleton */}
      {phase === 'scoring' && (
        <div className="space-y-2">
          {Array.from({ length: Math.min(rows.length, 5) }).map((_, i) => (
            <div key={i} className="skeleton h-12 rounded-lg" />
          ))}
        </div>
      )}

      {/* Results */}
      {phase === 'done' && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-900">
              Scored {scored.length} row{scored.length !== 1 ? 's' : ''}
              {skipped > 0 && <span className="font-normal text-slate-500"> · {skipped} skipped</span>}
            </h3>
            <button onClick={reset} className="text-sm font-medium text-indigo-600 hover:text-indigo-700">
              Start over
            </button>
          </div>
          {scored.length > 0 ? (
            <ResultsTable rows={scored} />
          ) : (
            <InlineNotice variant="warning" title="No rows could be scored" message="Every row was skipped by the judge. Check the CSV contents and try again." />
          )}
        </div>
      )}
    </div>
  );
}
