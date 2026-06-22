import type { Metadata } from "next";
import { Archivo } from "next/font/google";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";
import SiteHeader from "@/components/SiteHeader";

/**
 * Display face for headings. next/font bundles Archivo into the build and
 * self-hosts it, so there is no runtime fetch.
 */
const archivo = Archivo({
  subsets: ["latin"],
  weight: ["600", "700"],
  variable: "--font-archivo",
  display: "swap",
});

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
    <html
      lang="en"
      className={`${archivo.variable} ${GeistSans.variable} ${GeistMono.variable}`}
    >
      <body className="min-h-screen antialiased">
        <SiteHeader />
        <main className="mx-auto max-w-7xl px-4 pb-20 pt-8 sm:px-6 lg:px-8">
          {children}
        </main>
        <footer className="border-t border-hairline">
          <div className="mx-auto max-w-7xl px-4 py-6 text-sm text-muted sm:px-6 lg:px-8">
            Stateless by design — no database. Baked sample data renders instantly; the live judge
            scores in-session only.
          </div>
        </footer>
      </body>
    </html>
  );
}
