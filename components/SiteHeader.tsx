'use client';

/**
 * App-wide header + primary navigation. One coherent chrome across all pages:
 * a tick-scale wordmark in ink + petrol, and a flat nav where the active tab is
 * marked with a petrol underline (no pills, no segmented control).
 */

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const NAV = [
  { href: '/', label: 'Dashboard' },
  { href: '/evaluate', label: 'Score Output' },
  { href: '/compare', label: 'A/B Compare' },
  { href: '/upload', label: 'Batch CSV' },
];

/** A minimal calibration-scale glyph: a ruler baseline with one petrol needle. */
function ScaleMark() {
  return (
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none" aria-hidden="true">
      <line x1="3" y1="18" x2="23" y2="18" stroke="#15191C" strokeWidth="1.6" strokeLinecap="round" />
      {[3, 7, 11, 15, 19, 23].map((x) => (
        <line key={x} x1={x} y1="18" x2={x} y2={x === 15 ? 18 : 14} stroke="#15191C" strokeWidth="1.4" strokeLinecap="round" />
      ))}
      {/* The measured value — a petrol needle at one tick. */}
      <line x1="15" y1="7" x2="15" y2="18" stroke="#0E6E73" strokeWidth="2" strokeLinecap="round" />
      <circle cx="15" cy="7" r="2" fill="#0E6E73" />
    </svg>
  );
}

export default function SiteHeader() {
  const pathname = usePathname();

  const isActive = (href: string) =>
    href === '/' ? pathname === '/' : pathname.startsWith(href);

  return (
    <header className="sticky top-0 z-20 border-b border-hairline bg-surface">
      <div className="mx-auto max-w-7xl px-3 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-1 py-3 sm:h-16 sm:flex-row sm:items-center sm:justify-between sm:gap-4 sm:py-0">
          <Link href="/" className="flex items-center gap-2.5">
            <ScaleMark />
            <span className="flex flex-col leading-none">
              <span className="text-[15px] font-semibold tracking-tight text-ink">Eval Bench</span>
              <span className="mt-0.5 text-[10px] font-medium uppercase tracking-[0.18em] text-muted">
                LLM-as-Judge
              </span>
            </span>
          </Link>

          <nav className="flex w-full items-center justify-between gap-1 overflow-x-auto sm:w-auto sm:justify-end sm:gap-2">
            {NAV.map((item) => {
              const active = isActive(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  aria-current={active ? 'page' : undefined}
                  className={`whitespace-nowrap border-b-2 px-1 py-1 text-xs font-medium transition-colors duration-150 sm:px-2.5 sm:text-sm ${
                    active
                      ? 'border-petrol text-petrol'
                      : 'border-transparent text-muted hover:text-ink'
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </div>
    </header>
  );
}
