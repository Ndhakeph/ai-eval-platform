import type { Metadata } from "next";
import "./globals.css";
import SiteHeader from "@/components/SiteHeader";

export const metadata: Metadata = {
  title: "AI Evaluation Platform — LLM-as-Judge",
  description:
    "Rubric scoring and bias-aware pairwise evaluation for LLM outputs. Stateless: baked sample data plus a live judge, no database.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <SiteHeader />
        <main className="mx-auto max-w-7xl px-4 pb-20 pt-8 sm:px-6 lg:px-8">
          {children}
        </main>
        <footer className="border-t border-slate-200 bg-white">
          <div className="mx-auto max-w-7xl px-4 py-6 text-sm text-slate-500 sm:px-6 lg:px-8">
            Stateless by design — no database. Baked sample data renders instantly; the live judge
            scores in-session only.
          </div>
        </footer>
      </body>
    </html>
  );
}
