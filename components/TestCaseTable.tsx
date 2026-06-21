'use client';

/**
 * Data-dense results table for a list of scored evaluations. Each row expands to
 * reveal the output and the judge's per-criterion reasoning. Used by the
 * dashboard and the batch-CSV results view.
 */

import { Fragment, useState } from 'react';
import { ScoredEvaluation } from '@/types';
import { scoreTone, formatScore } from '@/lib/score-format';
import ScorePill from './ScorePill';

function HeaderCell({ children, align = 'left' }: { children: React.ReactNode; align?: 'left' | 'center' }) {
  return (
    <th
      className={`px-4 py-3 text-xs font-semibold uppercase tracking-wide text-slate-500 ${
        align === 'center' ? 'text-center' : 'text-left'
      }`}
    >
      {children}
    </th>
  );
}

export default function ResultsTable({ rows }: { rows: ScoredEvaluation[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (rows.length === 0) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-8 text-center text-sm text-slate-500">
        No evaluations to show yet.
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-200">
          <thead className="bg-slate-50">
            <tr>
              <HeaderCell>Domain</HeaderCell>
              <HeaderCell>Prompt</HeaderCell>
              <HeaderCell align="center">Acc.</HeaderCell>
              <HeaderCell align="center">Clar.</HeaderCell>
              <HeaderCell align="center">Comp.</HeaderCell>
              <HeaderCell align="center">Total</HeaderCell>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map((row) => {
              const open = expanded === row.id;
              const tone = scoreTone(row.total_score);
              return (
                <Fragment key={row.id}>
                  <tr
                    onClick={() => setExpanded(open ? null : row.id)}
                    className="cursor-pointer transition-colors hover:bg-slate-50"
                  >
                    <td className="whitespace-nowrap px-4 py-3">
                      <span className="inline-flex rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
                        {row.domain}
                      </span>
                    </td>
                    <td className="max-w-md px-4 py-3 text-sm text-slate-700">
                      <span className="line-clamp-1">{row.prompt}</span>
                    </td>
                    <td className="px-4 py-3 text-center"><ScorePill score={row.accuracy.score} size="sm" /></td>
                    <td className="px-4 py-3 text-center"><ScorePill score={row.clarity.score} size="sm" /></td>
                    <td className="px-4 py-3 text-center"><ScorePill score={row.completeness.score} size="sm" /></td>
                    <td className={`px-4 py-3 text-center text-sm font-bold tabular-nums ${tone.text}`}>
                      {formatScore(row.total_score)}
                    </td>
                  </tr>
                  {open && (
                    <tr className="bg-slate-50/60">
                      <td colSpan={6} className="px-4 py-4">
                        <div className="space-y-3 text-sm">
                          <div>
                            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Output</p>
                            <p className="mt-1 whitespace-pre-wrap rounded-lg bg-white p-3 font-mono text-xs text-slate-700 ring-1 ring-slate-200">
                              {row.output}
                            </p>
                          </div>
                          <div className="grid gap-3 sm:grid-cols-3">
                            {(['accuracy', 'clarity', 'completeness'] as const).map((c) => (
                              <div key={c} className="rounded-lg bg-white p-3 ring-1 ring-slate-200">
                                <div className="flex items-center justify-between">
                                  <span className="text-xs font-semibold capitalize text-slate-600">{c}</span>
                                  <ScorePill score={row[c].score} size="sm" />
                                </div>
                                <p className="mt-1.5 text-xs leading-relaxed text-slate-600">{row[c].reasoning}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
