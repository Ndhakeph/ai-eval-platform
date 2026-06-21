'use client';

/**
 * App-wide header + primary navigation. One coherent chrome across all pages,
 * with the active tab highlighted via the current pathname.
 */

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const NAV = [
  { href: '/', label: 'Dashboard' },
  { href: '/evaluate', label: 'Score Output' },
  { href: '/compare', label: 'A/B Compare' },
  { href: '/upload', label: 'Batch CSV' },
];

export default function SiteHeader() {
  const pathname = usePathname();

  const isActive = (href: string) =>
    href === '/' ? pathname === '/' : pathname.startsWith(href);

  return (
    <header className="sticky top-0 z-20 border-b border-slate-900/[0.06] bg-white/75 backdrop-blur-xl">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between gap-4">
          <Link href="/" className="group flex items-center gap-2.5">
            <span className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-600 to-violet-600 shadow-[0_6px_16px_-6px_rgba(124,58,237,0.7)] transition-transform duration-200 group-hover:scale-105">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M4 14.5 9 19.5 20 6.5" stroke="white" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M4 8.5 6.2 10.7" stroke="white" strokeOpacity="0.55" strokeWidth="2.2" strokeLinecap="round" />
              </svg>
            </span>
            <span className="flex flex-col leading-none">
              <span className="text-[15px] font-semibold tracking-tight text-slate-900">Eval Platform</span>
              <span className="mt-0.5 text-[10px] font-medium uppercase tracking-[0.18em] text-slate-400">
                LLM-as-Judge
              </span>
            </span>
          </Link>

          <nav className="flex items-center gap-0.5 rounded-full bg-slate-100/80 p-1 ring-1 ring-slate-900/[0.04]">
            {NAV.map((item) => {
              const active = isActive(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  aria-current={active ? 'page' : undefined}
                  className={`rounded-full px-3.5 py-1.5 text-sm font-medium transition-all duration-200 ${
                    active
                      ? 'bg-white text-indigo-700 shadow-[0_1px_2px_rgba(15,23,42,0.06),0_4px_10px_-6px_rgba(15,23,42,0.25)]'
                      : 'text-slate-500 hover:text-slate-900'
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
