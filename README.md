# AI Eval Platform

Automated quality scoring for AI-generated content using the LLM-as-Judge pattern — upload test cases via CSV, run rubric-based evaluations with GPT-4o-mini via FastRouter, and track scores on a dashboard.

<!-- [![Live Demo](https://img.shields.io/badge/Live_Demo-▶_Try_It-blue?style=for-the-badge)](DEPLOY_URL) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## Why I Built This

I wanted a way to systematically measure whether an LLM's output is actually good — not just "looks fine" but scored against a rubric. Manual review doesn't scale, and metrics like BLEU/ROUGE miss semantic meaning entirely. I built this platform so you can upload a CSV of prompt/expected-output pairs, have an LLM judge the actual outputs on accuracy, clarity, and completeness (0-10 each), and see the results in a dashboard with trend charts. The whole thing runs on free/cheap tiers — Supabase for storage, GPT-4o-mini via FastRouter for evaluation, Vercel for hosting.

## How It Works

```mermaid
flowchart LR
    A[CSV Upload] --> B[Supabase DB]
    B --> C[Evaluation Engine]
    C --> D[FastRouter API\nLLM-as-Judge]
    D --> E[Rubric Scores\nAccuracy · Clarity · Completeness]
    E --> F[Dashboard\n& Trend Charts]
```

1. **Upload** — Drop a CSV with `prompt` and `expected_output` columns. The platform validates and stores them in Supabase.
2. **Evaluate** — For each test case, the LLM generates an actual output, then a second call acts as judge — scoring accuracy, clarity, and completeness against the expected output using a structured rubric prompt.
3. **Analyze** — The dashboard shows per-evaluation scores, aggregate averages, and trend lines over time. Export results as CSV for further analysis.

The judge prompt uses tiered scoring criteria (0-3 poor, 4-6 partial, 7-8 good, 9-10 excellent) and returns structured JSON, which gets validated and stored. Batch evaluations run sequentially with 1-second delays to respect rate limits.

## Key Technical Decisions

- **Same model as both generator and judge** — I use the same model to generate outputs and then judge them in a separate call. This is intentional: the platform evaluates any model's output against a rubric. The dual-call pattern (generate → judge) keeps the evaluation pipeline modular, and the model is configurable via environment variable so I can swap providers without touching code.

- **Sequential batch processing over parallel** — Batch evaluations process one at a time with a 1-second delay between API calls. This trades speed for reliability — no rate limit failures, no partial batches, and each test case gets the full API context window. For 100 test cases, this takes ~3-4 minutes, which is acceptable for an evaluation workflow.

- **Structured JSON rubric response** — The judge prompt forces a strict `{"accuracy": N, "clarity": N, "completeness": N}` JSON response with no explanation text. I strip markdown code blocks if the LLM wraps the response, then validate all three scores are 0-10 integers. This makes the scoring pipeline deterministic and parseable without any NLP post-processing.

## Tech Stack

- **Next.js 16** — App Router, API routes, React Server Components
- **TypeScript** — Strict types for test cases, scores, and evaluation results
- **Supabase** — PostgreSQL backend for test cases and evaluation history
- **FastRouter** — OpenAI-compatible API routing, defaulting to `openai/gpt-4o-mini`
- **Recharts** — Score trend visualization on the results dashboard
- **Tailwind CSS** — Utility-first styling
- **Papa Parse** — CSV parsing with validation

## Getting Started

```bash
git clone https://github.com/tacitusblindsbig/ai-eval-platform.git
cd ai-eval-platform
npm install
cp .env.example .env.local
# Fill in your keys in .env.local
npm run dev
```

You'll need:
- A [FastRouter API key](https://fastrouter.ai) (budget-friendly API routing)
- A [Supabase](https://supabase.com) project — run this SQL in the SQL Editor:

```sql
CREATE TABLE test_cases (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  prompt TEXT NOT NULL,
  expected_output TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE evaluation_results (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  test_case_id UUID REFERENCES test_cases(id) ON DELETE CASCADE,
  actual_output TEXT NOT NULL,
  accuracy_score INTEGER CHECK (accuracy_score >= 0 AND accuracy_score <= 10),
  clarity_score INTEGER CHECK (clarity_score >= 0 AND clarity_score <= 10),
  completeness_score INTEGER CHECK (completeness_score >= 0 AND completeness_score <= 10),
  total_score INTEGER CHECK (total_score >= 0 AND total_score <= 30),
  model_used TEXT DEFAULT 'openai/gpt-4o-mini',
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_eval_test_case ON evaluation_results(test_case_id);
CREATE INDEX idx_eval_created_at ON evaluation_results(created_at DESC);
```

## Project Structure

```
├── app/
│   ├── api/          # evaluate, upload, results, stats, test-cases
│   ├── evaluate/     # Run evaluations page
│   ├── results/      # Results dashboard with charts
│   ├── upload/       # CSV upload page
│   └── page.tsx      # Dashboard homepage
├── components/       # EvalCard, ScoreChart, TestCaseTable, UploadForm
├── lib/              # llm.ts, evaluator.ts, supabase.ts, csv-parser.ts
└── types/            # TypeScript interfaces
```

## What I Learned

Building this taught me that the hardest part of LLM-as-Judge isn't the API call — it's designing a rubric prompt that produces consistent, meaningful scores. My first version returned wildly different scores for the same input because the criteria were too vague. Adding tiered scoring bands (0-3, 4-6, 7-8, 9-10) with specific descriptions for each level made the outputs dramatically more stable. I also learned that forcing JSON-only responses from LLMs requires defensive parsing — models occasionally wrap output in markdown code blocks even when explicitly told not to, so I had to add a stripping layer. Switching from Gemini to an OpenAI-compatible provider (FastRouter) was straightforward because the evaluation logic was already decoupled from the API client — the model name is now just an env var. The batch processing design was a practical lesson in choosing reliability over throughput: sequential execution with delays is boring but it never fails halfway through a 100-item batch.
