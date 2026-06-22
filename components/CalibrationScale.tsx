/**
 * The signature element: a calibration scale. A score is rendered as a precise
 * position on a 0–10 horizontal track with tick marks at each integer (a
 * vernier / instrument scale). The value is large in Geist Mono and coloured by
 * the score-scale semantic. This replaces the radial gauge entirely.
 *
 * Pure presentation (no hooks), so it renders on both server and client. The
 * one bit of motion — the fill settling to its value — is CSS-only and runs
 * once on mount when a `cal-animate` ancestor is present.
 */

import { scoreHex, scoreBand, formatScore } from '@/lib/score-format';

const TICKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const LABELS = [0, 2, 4, 6, 8, 10];

function clampPct(score: number): number {
  return Math.max(0, Math.min(100, (score / 10) * 100));
}

/** The 0–10 vernier track: hairline ticks, a coloured fill, and a needle. */
function Track({
  pct,
  hex,
  height,
  showLabels = false,
}: {
  pct: number;
  hex: string;
  height: number;
  showLabels?: boolean;
}) {
  return (
    <div>
      <div className="relative" style={{ height }}>
        {/* Baseline */}
        <div className="absolute inset-x-0 bottom-0 h-px bg-hairline" />
        {/* Integer ticks — taller on even integers (major), shorter on odd. */}
        {TICKS.map((t) => (
          <div
            key={t}
            className="absolute bottom-0 w-px bg-hairline"
            style={{ left: `${t * 10}%`, height: t % 2 === 0 ? 12 : 7 }}
          />
        ))}
        {/* Measured value: coloured fill to the score position… */}
        <div
          className="cal-fill absolute bottom-0 h-[3px]"
          style={{ width: `${pct}%`, backgroundColor: hex }}
        />
        {/* …and a needle pointing at it. */}
        <div
          className="cal-needle absolute bottom-0 w-[2px]"
          style={{ left: `${pct}%`, height, marginLeft: -1, backgroundColor: hex }}
        />
      </div>
      {showLabels && (
        <div className="relative mt-1.5 h-3">
          {LABELS.map((l) => (
            <span
              key={l}
              className="absolute -translate-x-1/2 font-mono text-[10px] tabular-nums text-muted"
              style={{ left: `${l * 10}%` }}
            >
              {l}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/** Large total readout: the headline value in mono plus the labelled track. */
export function CalibrationReadout({
  score,
  caption = 'Total score',
}: {
  score: number;
  caption?: string;
}) {
  const hex = scoreHex(score);
  return (
    <div>
      <div className="flex items-end gap-3">
        <span
          className="font-mono text-5xl font-semibold leading-none tabular-nums"
          style={{ color: hex }}
        >
          {formatScore(score)}
        </span>
        <span className="mb-1 text-sm text-muted">{caption} · 0–10</span>
        <span
          className="mb-1 ml-auto text-[11px] font-semibold uppercase tracking-[0.14em]"
          style={{ color: hex }}
        >
          {scoreBand(score)}
        </span>
      </div>
      <div className="mt-3">
        <Track pct={clampPct(score)} hex={hex} height={44} showLabels />
      </div>
    </div>
  );
}

/** One rubric criterion: a compact calibrated bar with its reasoning beside it. */
export function CriterionBar({
  label,
  score,
  reasoning,
}: {
  label: string;
  score: number;
  reasoning: string;
}) {
  const hex = scoreHex(score);
  return (
    <div className="py-3">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-sm font-medium text-ink">{label}</span>
        <span className="font-mono text-sm font-semibold tabular-nums" style={{ color: hex }}>
          {formatScore(score)}
          <span className="text-muted">/10</span>
        </span>
      </div>
      <div className="mt-2">
        <Track pct={clampPct(score)} hex={hex} height={20} />
      </div>
      <p className="mt-2 text-sm leading-relaxed text-muted">{reasoning}</p>
    </div>
  );
}
