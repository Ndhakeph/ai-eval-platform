/**
 * A calm inline notice for degraded states — missing API key, rate limit, or a
 * judge error. Never a stack trace. Optionally renders a call-to-action (e.g.
 * "view the baked sample instead").
 */

type Variant = 'info' | 'warning' | 'error';

const STYLES: Record<Variant, { box: string; icon: string; symbol: string }> = {
  info: { box: 'border-indigo-200 bg-indigo-50 text-indigo-900', icon: 'text-indigo-500', symbol: 'ℹ' },
  warning: { box: 'border-amber-200 bg-amber-50 text-amber-900', icon: 'text-amber-500', symbol: '!' },
  error: { box: 'border-rose-200 bg-rose-50 text-rose-900', icon: 'text-rose-500', symbol: '×' },
};

export default function InlineNotice({
  variant = 'info',
  title,
  message,
  action,
}: {
  variant?: Variant;
  title: string;
  message?: string;
  action?: React.ReactNode;
}) {
  const s = STYLES[variant];
  return (
    <div className={`rounded-2xl border p-4 ${s.box}`}>
      <div className="flex items-start gap-3">
        <span className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-white text-sm font-bold ${s.icon}`}>
          {s.symbol}
        </span>
        <div className="flex-1">
          <p className="text-sm font-semibold">{title}</p>
          {message && <p className="mt-1 text-sm opacity-90">{message}</p>}
          {action && <div className="mt-3">{action}</div>}
        </div>
      </div>
    </div>
  );
}
