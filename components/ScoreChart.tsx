'use client';

/**
 * Dashboard visualizations (Recharts), drawn as instrument readouts: solid
 * score-scale fills (no gradients), hairline axes, tabular-mono value labels,
 * and a plain light tooltip. Three focused charts — criterion / distribution /
 * domain — all driven by the baked sample data. (The radial gauge has been
 * removed; the calibration scale is the score readout now.)
 */

import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LabelList,
} from 'recharts';
import { scoreHex } from '@/lib/score-format';

const AXIS = { fontSize: 11, fill: '#6A6F6B', fontWeight: 500 } as const;
const GRID = '#DEDFD9';
const CURSOR = { fill: 'rgba(21,25,28,0.04)' };
const MONO = 'var(--font-geist-mono), ui-monospace, monospace';

interface TooltipPayload {
  value: number;
  payload: Record<string, unknown>;
}

function BenchTooltip({
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
    <div className="rounded-md border border-hairline bg-surface px-3 py-2 shadow-[0_8px_24px_-16px_rgba(21,25,28,0.4)]">
      {label && <p className="text-[11px] font-medium text-muted">{label}</p>}
      <p className="text-sm font-semibold text-ink">
        {valueLabel}:{' '}
        <span className="font-mono tabular-nums">{Number(payload[0].value).toFixed(digits)}</span>
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
      <h3 className="text-sm font-semibold text-ink">{title}</h3>
      <p className="mt-0.5 text-xs text-muted">{subtitle}</p>
      <div className="mt-4">{children}</div>
    </div>
  );
}

const labelStyle = { fill: '#15191C', fontSize: 11, fontWeight: 600, fontFamily: MONO };

/** LabelList formatters receive a broad ReactNode-ish type, so coerce safely. */
const fmt1 = (v: React.ReactNode) => Number(typeof v === 'number' ? v : 0).toFixed(1);

/** Colour a distribution band by the score range it represents. */
function bandHex(band: string): string {
  const lo = Number(band.split('-')[0]);
  return scoreHex(lo >= 8 ? 9 : lo >= 4 ? 6 : 2);
}

export function CriterionAverageChart({ data }: { data: { criterion: string; average: number }[] }) {
  return (
    <ChartShell title="Average score by criterion" subtitle="Mean across all sample evaluations (0–10)">
      <ResponsiveContainer width="100%" height={228}>
        <BarChart data={data} margin={{ top: 18, right: 8, bottom: 0, left: -18 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
          <XAxis dataKey="criterion" tick={AXIS} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 10]} tick={AXIS} axisLine={false} tickLine={false} />
          <Tooltip cursor={CURSOR} content={<BenchTooltip valueLabel="Average" />} />
          <Bar dataKey="average" radius={[3, 3, 0, 0]} maxBarSize={68}>
            {data.map((d) => (
              <Cell key={d.criterion} fill={scoreHex(d.average)} />
            ))}
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
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
          <XAxis dataKey="band" tick={AXIS} axisLine={false} tickLine={false} />
          <YAxis allowDecimals={false} tick={AXIS} axisLine={false} tickLine={false} />
          <Tooltip cursor={CURSOR} content={<BenchTooltip valueLabel="Evaluations" digits={0} />} />
          <Bar dataKey="count" radius={[3, 3, 0, 0]} maxBarSize={60}>
            {data.map((d) => (
              <Cell key={d.band} fill={bandHex(d.band)} />
            ))}
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
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
          <XAxis type="number" domain={[0, 10]} tick={AXIS} axisLine={false} tickLine={false} />
          <YAxis
            type="category"
            dataKey="domain"
            tick={{ ...AXIS, fill: '#15191C' }}
            axisLine={false}
            tickLine={false}
            width={108}
          />
          <Tooltip cursor={CURSOR} content={<BenchTooltip valueLabel="Avg score" />} />
          <Bar dataKey="average" radius={[0, 3, 3, 0]} maxBarSize={24}>
            {data.map((d) => (
              <Cell key={d.domain} fill={scoreHex(d.average)} />
            ))}
            <LabelList dataKey="average" position="right" formatter={fmt1} style={labelStyle} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}
