/**
 * A calm inline notice for degraded states — missing API key, rate limit, or a
 * judge error. Never a stack trace. The accent is a quiet left rule in petrol
 * (info), ochre (warning), or rust (error); copy gives direction, not mood.
 * Optionally renders a call-to-action (e.g. "view the baked sample instead").
 */

type Variant = 'info' | 'warning' | 'error';

const ACCENT: Record<Variant, string> = {
  info: '#0E6E73',
  warning: '#B5862B',
  error: '#B14430',
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
  return (
    <div
      className="rounded-md border border-l-4 border-hairline bg-surface p-4"
      style={{ borderLeftColor: ACCENT[variant] }}
    >
      <p className="text-sm font-semibold text-ink">{title}</p>
      {message && <p className="mt-1 text-sm text-muted">{message}</p>}
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}
