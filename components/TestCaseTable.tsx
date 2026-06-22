'use client';

/**
 * Data-dense results table for a list of scored evaluations. Each row expands to
 * reveal the output and the judge's per-criterion reasoning. Used by the
 * dashboard and the batch-CSV results view. Figures are mono + tabular so the
 * score columns align like an instrument readout.
 */

import { Fragment, useState } from 'react';
import { ScoredEvaluation } from '@/types';
import { scoreTone, formatScore } from '@/lib/score-format';
import ScorePill from './ScorePill';

function HeaderCell({ children, align = 'left' }: { children: React.ReactNode; align?: 'left' | 'center' }) {
  return (
    <th
      className={`px-4 py-3 text-xs font-semibold uppercase tracking-wide text-muted ${
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
      <div className="card p-8 text-center text-sm text-muted">
        No evaluations to show yet.
      </div>
    );
  }

  return (
    <div className="card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-hairline">
          <thead className="bg-paper">
            <tr>
              <HeaderCell>Domain</HeaderCell>
              <HeaderCell>Prompt</HeaderCell>
              <HeaderCell align="center">Acc.</HeaderCell>
              <HeaderCell align="center">Clar.</HeaderCell>
              <HeaderCell align="center">Comp.</HeaderCell>
              <HeaderCell align="center">Total</HeaderCell>
            </tr>
          </thead>
          <tbody className="divide-y divide-hairline">
            {rows.map((row) => {
              const open = expanded === row.id;
              const tone = scoreTone(row.total_score);
              return (
                <Fragment key={row.id}>
                  <tr
                    onClick={() => setExpanded(open ? null : row.id)}
                    className="cursor-pointer transition-colors hover:bg-paper"
                  >
                    <td className="whitespace-nowrap px-4 py-3">
                      <span className="inline-flex rounded border border-hairline bg-paper px-2 py-0.5 text-xs font-medium text-muted">
                        {row.domain}
                      </span>
                    </td>
                    <td className="max-w-md px-4 py-3 text-sm text-ink">
                      <span className="line-clamp-1">{row.prompt}</span>
                    </td>
                    <td className="px-4 py-3 text-center"><ScorePill score={row.accuracy.score} size="sm" /></td>
                    <td className="px-4 py-3 text-center"><ScorePill score={row.clarity.score} size="sm" /></td>
                    <td className="px-4 py-3 text-center"><ScorePill score={row.completeness.score} size="sm" /></td>
                    <td className={`px-4 py-3 text-center font-mono text-sm font-bold tabular-nums ${tone.text}`}>
                      {formatScore(row.total_score)}
                    </td>
                  </tr>
                  {open && (
                    <tr className="bg-paper">
                      <td colSpan={6} className="px-4 py-4">
                        <div className="space-y-3 text-sm">
                          <div>
                            <p className="text-xs font-semibold uppercase tracking-wide text-muted">Output</p>
                            <p className="mt-1 whitespace-pre-wrap rounded-md border border-hairline bg-surface p-3 font-mono text-xs text-ink">
                              {row.output}
                            </p>
                          </div>
                          <div className="grid gap-3 sm:grid-cols-3">
                            {(['accuracy', 'clarity', 'completeness'] as const).map((c) => (
                              <div key={c} className="rounded-md border border-hairline bg-surface p-3">
                                <div className="flex items-center justify-between">
                                  <span className="text-xs font-semibold capitalize text-ink">{c}</span>
                                  <ScorePill score={row[c].score} size="sm" />
                                </div>
                                <p className="mt-1.5 text-xs leading-relaxed text-muted">{row[c].reasoning}</p>
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
