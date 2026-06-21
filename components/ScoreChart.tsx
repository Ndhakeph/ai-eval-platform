'use client';

/**
 * Dashboard visualizations (Recharts). Three small, focused charts rather than
 * one busy one: criterion averages, score distribution, and per-domain average.
 * All driven by the baked sample data — no fetching.
 */

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { scoreTone } from '@/lib/score-format';

const AXIS = { fontSize: 12, fill: '#64748b' };
const GRID = '#e2e8f0';
const INDIGO = '#4f46e5';

const tooltipStyle = {
  contentStyle: {
    backgroundColor: 'white',
    border: '1px solid #e2e8f0',
    borderRadius: '0.5rem',
    fontSize: '0.8rem',
    boxShadow: '0 4px 12px rgba(15,23,42,0.08)',
  },
  cursor: { fill: 'rgba(79,70,229,0.06)' },
};

function ChartShell({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
      <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>
      <div className="mt-4">{children}</div>
    </div>
  );
}

export function CriterionAverageChart({ data }: { data: { criterion: string; average: number }[] }) {
  return (
    <ChartShell title="Average score by criterion" subtitle="Mean across all sample evaluations (0-10)">
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
          <XAxis dataKey="criterion" tick={AXIS} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 10]} tick={AXIS} axisLine={false} tickLine={false} />
          <Tooltip {...tooltipStyle} formatter={(value: number) => [value.toFixed(2), 'Average']} />
          <Bar dataKey="average" radius={[6, 6, 0, 0]} maxBarSize={64}>
            {data.map((entry) => (
              <Cell key={entry.criterion} fill={scoreTone(entry.average).hex} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function ScoreDistributionChart({ data }: { data: { band: string; count: number }[] }) {
  return (
    <ChartShell title="Score distribution" subtitle="How many evaluations fall in each total-score band">
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
          <XAxis dataKey="band" tick={AXIS} axisLine={false} tickLine={false} />
          <YAxis allowDecimals={false} tick={AXIS} axisLine={false} tickLine={false} />
          <Tooltip {...tooltipStyle} formatter={(value: number) => [value, 'Evaluations']} />
          <Bar dataKey="count" fill={INDIGO} radius={[6, 6, 0, 0]} maxBarSize={56} />
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function DomainAverageChart({ data }: { data: { domain: string; average: number }[] }) {
  return (
    <ChartShell title="Average score by domain" subtitle="Mean total score per subject area">
      <ResponsiveContainer width="100%" height={Math.max(220, data.length * 34)}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
          <XAxis type="number" domain={[0, 10]} tick={AXIS} axisLine={false} tickLine={false} />
          <YAxis
            type="category"
            dataKey="domain"
            tick={AXIS}
            axisLine={false}
            tickLine={false}
            width={104}
          />
          <Tooltip {...tooltipStyle} formatter={(value: number) => [value.toFixed(2), 'Avg score']} />
          <Bar dataKey="average" radius={[0, 6, 6, 0]} maxBarSize={22}>
            {data.map((entry) => (
              <Cell key={entry.domain} fill={scoreTone(entry.average).hex} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}
