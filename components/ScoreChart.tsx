'use client';

/**
 * Dashboard visualizations (Recharts), tuned for a crafted look: accent-gradient
 * fills, value labels, a dark floating tooltip, and hairline axes. Four focused
 * charts — a radial score gauge plus criterion / distribution / domain bars —
 * all driven by the baked sample data.
 */

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LabelList,
  RadialBarChart,
  RadialBar,
  PolarAngleAxis,
} from 'recharts';

const AXIS = { fontSize: 11, fill: '#94a3b8', fontWeight: 500 };
const GRID = '#eef2f6';

interface TooltipPayload {
  value: number;
  payload: Record<string, unknown>;
}

function FloatingTooltip({
  active,
  payload,
  label,
  valueLabel,
  digits = 2,
}: {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: string;
  valueLabel: string;
  digits?: number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl bg-slate-900 px-3 py-2 shadow-[0_8px_24px_-6px_rgba(15,23,42,0.5)]">
      {label && <p className="text-[11px] font-medium text-slate-400">{label}</p>}
      <p className="text-sm font-semibold text-white">
        {valueLabel}: {Number(payload[0].value).toFixed(digits)}
      </p>
    </div>
  );
}

function ChartShell({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div className="card p-5">
      <h3 className="text-sm font-semibold tracking-tight text-slate-900">{title}</h3>
      <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>
      <div className="mt-4">{children}</div>
    </div>
  );
}

const accentGradient = (id: string, vertical = true) => (
  <linearGradient id={id} x1="0" y1="0" x2={vertical ? '0' : '1'} y2={vertical ? '1' : '0'}>
    <stop offset="0%" stopColor="#6366f1" />
    <stop offset="100%" stopColor="#8b5cf6" />
  </linearGradient>
);

const labelStyle = { fill: '#475569', fontSize: 11, fontWeight: 600 };

/** LabelList formatters receive a broad ReactNode-ish type, so coerce safely. */
const fmt1 = (v: React.ReactNode) => Number(typeof v === 'number' ? v : 0).toFixed(1);

/** Radial score gauge — the dashboard hero element. */
export function ScoreGauge({ value, caption }: { value: number; caption: string }) {
  return (
    <div className="relative mx-auto h-44 w-44">
      <ResponsiveContainer width="100%" height="100%">
        <RadialBarChart
          innerRadius="74%"
          outerRadius="100%"
          data={[{ value }]}
          startAngle={90}
          endAngle={-270}
        >
          <defs>{accentGradient('gaugeGrad')}</defs>
          <PolarAngleAxis type="number" domain={[0, 10]} angleAxisId={0} tick={false} />
          <RadialBar
            background={{ fill: '#eef2f6' }}
            dataKey="value"
            cornerRadius={20}
            angleAxisId={0}
            fill="url(#gaugeGrad)"
          />
        </RadialBarChart>
      </ResponsiveContainer>
      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-bold tracking-tight text-slate-900 tabular-nums">{value.toFixed(2)}</span>
        <span className="mt-1 text-[11px] font-medium uppercase tracking-[0.14em] text-slate-400">{caption}</span>
      </div>
    </div>
  );
}

export function CriterionAverageChart({ data }: { data: { criterion: string; average: number }[] }) {
  return (
    <ChartShell title="Average score by criterion" subtitle="Mean across all sample evaluations (0–10)">
      <ResponsiveContainer width="100%" height={228}>
        <BarChart data={data} margin={{ top: 18, right: 8, bottom: 0, left: -18 }}>
          <defs>{accentGradient('critGrad')}</defs>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
          <XAxis dataKey="criterion" tick={AXIS} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 10]} tick={AXIS} axisLine={false} tickLine={false} />
          <Tooltip
            cursor={{ fill: 'rgba(99,102,241,0.06)' }}
            content={<FloatingTooltip valueLabel="Average" />}
          />
          <Bar dataKey="average" radius={[8, 8, 0, 0]} maxBarSize={68} fill="url(#critGrad)">
            <LabelList dataKey="average" position="top" formatter={fmt1} style={labelStyle} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function ScoreDistributionChart({ data }: { data: { band: string; count: number }[] }) {
  return (
    <ChartShell title="Score distribution" subtitle="How many evaluations fall in each total-score band">
      <ResponsiveContainer width="100%" height={228}>
        <BarChart data={data} margin={{ top: 18, right: 8, bottom: 0, left: -18 }}>
          <defs>{accentGradient('distGrad')}</defs>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
          <XAxis dataKey="band" tick={AXIS} axisLine={false} tickLine={false} />
          <YAxis allowDecimals={false} tick={AXIS} axisLine={false} tickLine={false} />
          <Tooltip
            cursor={{ fill: 'rgba(99,102,241,0.06)' }}
            content={<FloatingTooltip valueLabel="Evaluations" digits={0} />}
          />
          <Bar dataKey="count" radius={[8, 8, 0, 0]} maxBarSize={60} fill="url(#distGrad)">
            <LabelList dataKey="count" position="top" style={labelStyle} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function DomainAverageChart({ data }: { data: { domain: string; average: number }[] }) {
  return (
    <ChartShell title="Average score by domain" subtitle="Mean total score per subject area">
      <ResponsiveContainer width="100%" height={Math.max(220, data.length * 38)}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 36, bottom: 0, left: 8 }}>
          <defs>{accentGradient('domainGrad', false)}</defs>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
          <XAxis type="number" domain={[0, 10]} tick={AXIS} axisLine={false} tickLine={false} />
          <YAxis
            type="category"
            dataKey="domain"
            tick={{ ...AXIS, fill: '#475569' }}
            axisLine={false}
            tickLine={false}
            width={108}
          />
          <Tooltip
            cursor={{ fill: 'rgba(99,102,241,0.06)' }}
            content={<FloatingTooltip valueLabel="Avg score" />}
          />
          <Bar dataKey="average" radius={[0, 8, 8, 0]} maxBarSize={24} fill="url(#domainGrad)">
            <LabelList dataKey="average" position="right" formatter={fmt1} style={labelStyle} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}
