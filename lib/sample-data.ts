/**
 * Pre-computed, hand-curated evaluation data — the default content of the app.
 *
 * The platform is stateless: there is no database and the dashboard makes no API
 * call. Everything you see on first load is rendered from this file. It is the
 * thing recruiters actually read, so the scores and reasoning are written to be
 * realistic, calibrated, and genuinely defensible.
 *
 * IMPORTANT: this module must stay dependency-free (types only) so it can be
 * imported directly into client components without dragging the OpenAI SDK into
 * the browser bundle. Totals are computed by a local helper, not `lib/judge`.
 */

import {
  ScoredEvaluation,
  ABComparison,
  OrderingResult,
  DashboardStats,
  CriterionDetail,
} from '@/types';

const MODEL = 'anthropic/claude-sonnet-4.6';

/** Real-decimal average, mirroring `calculateTotalScore` in lib/judge.ts. */
function avg3(a: number, c: number, co: number): number {
  return Math.round(((a + c + co) / 3) * 100) / 100;
}

type EvalSeed = Omit<ScoredEvaluation, 'total_score' | 'model_used'>;

function buildEval(seed: EvalSeed): ScoredEvaluation {
  return {
    ...seed,
    total_score: avg3(seed.accuracy.score, seed.clarity.score, seed.completeness.score),
    model_used: MODEL,
  };
}

const d = (score: number, reasoning: string): CriterionDetail => ({ score, reasoning });

/* ========================================================================== */
/* 1. Single-output scoring showcase (rich per-criterion reasoning)           */
/* ========================================================================== */

export const singleScoringExamples: ScoredEvaluation[] = [
  buildEval({
    id: 'single-grounding-refund',
    domain: 'RAG · Grounding',
    prompt:
      'Using only the context below, answer the question.\n\nContext: "Returns are accepted within 30 days of delivery for unused items in original packaging. Refunds are issued to the original payment method within 5 business days of the returned item being received."\n\nQuestion: What is the refund window, and how is the refund paid?',
    reference:
      'Returns are accepted within 30 days of delivery (unused, original packaging). The refund is issued to the original payment method within 5 business days of the return being received.',
    output:
      'You can request a refund within 60 days of delivery, and the amount is credited back to your original payment method within 5 business days of us receiving the item.',
    accuracy: d(
      3,
      'Not grounded in the context: the passage states a 30-day return window, but the answer asserts 60 days — a direct contradiction of the source. The payment-method and 5-business-day details are correct, so this is a confident, half-right answer, which is the dangerous kind.',
    ),
    clarity: d(
      9,
      'Fluent, well-structured, and reads with total confidence — precisely why an ungrounded number slips past a human skim.',
    ),
    completeness: d(
      7,
      'Addresses both parts of the question (window and payment method), but anchors the headline figure to a value that is not in the provided context.',
    ),
  }),
  buildEval({
    id: 'single-coding-palindrome',
    domain: 'Coding',
    prompt:
      'Write a Python function `is_palindrome(s)` that returns True if a string is a palindrome, ignoring case and non-alphanumeric characters.',
    reference:
      "def is_palindrome(s):\n    cleaned = [c.lower() for c in s if c.isalnum()]\n    return cleaned == cleaned[::-1]",
    output:
      "def is_palindrome(s):\n    filtered = ''.join(ch.lower() for ch in s if ch.isalnum())\n    return filtered == filtered[::-1]\n\n# Examples:\n# is_palindrome('A man, a plan, a canal: Panama') -> True\n# is_palindrome('race a car') -> False",
    accuracy: d(
      10,
      'The logic is correct: it lowercases, filters to alphanumerics with isalnum(), and compares the string to its reverse. Both worked examples evaluate correctly.',
    ),
    clarity: d(
      9,
      'Idiomatic and readable — a single generator expression plus a slice reversal. Variable naming is clear; the only nit is no docstring.',
    ),
    completeness: d(
      9,
      'Fully solves the task and even includes usage examples. Does not mention the empty-string edge case (which correctly returns True), hence not a perfect 10.',
    ),
  }),
  buildEval({
    id: 'single-medical-metformin',
    domain: 'Medical',
    prompt: 'A patient asks: what are the common side effects of metformin?',
    reference:
      'The most common side effects are gastrointestinal: nausea, diarrhea, abdominal discomfort, and a metallic taste, usually mild and improving over time. Long-term use can lower vitamin B12. Lactic acidosis is a rare but serious risk, especially with kidney impairment. Advise consulting a clinician.',
    output:
      'Metformin commonly causes gastrointestinal side effects such as nausea, diarrhea, and stomach upset, which often ease after the first few weeks or when taken with food. A rare but serious risk is lactic acidosis. You should always check with your doctor or pharmacist for advice specific to you.',
    accuracy: d(
      9,
      'Everything stated is medically correct: GI effects are the common ones, they ease with food/time, and lactic acidosis is correctly framed as rare but serious.',
    ),
    clarity: d(
      9,
      'Plain, patient-friendly language with an appropriate safety disclaimer. Well organized from common to rare.',
    ),
    completeness: d(
      6,
      'Omits two reference points: long-term vitamin B12 depletion and the metallic-taste side effect. The core answer is present but not comprehensive.',
    ),
  }),
  buildEval({
    id: 'single-legal-consideration',
    domain: 'Legal',
    prompt: "Explain in plain English what 'consideration' means in contract law.",
    reference:
      'Consideration is something of value that each party gives or promises in exchange for the other party’s promise — it is the "price" of the bargain. Without consideration on both sides, a promise is generally not an enforceable contract. It need not be money or be adequate, but it must be sufficient and bargained-for.',
    output:
      "Consideration is what each side brings to a contract — basically the value being exchanged. If I promise to sell you my bike for $100, my consideration is the bike and yours is the money. A promise without any consideration in return usually isn't a binding contract.",
    accuracy: d(
      9,
      'Correctly captures consideration as the bargained-for value exchanged and that a promise lacking it is usually unenforceable. The example is apt.',
    ),
    clarity: d(
      10,
      'Excellent plain-English explanation with a concrete, relatable example — exactly what the prompt asked for.',
    ),
    completeness: d(
      6,
      'Misses the nuance that consideration need not be adequate but must be sufficient. Good for a layperson but legally incomplete.',
    ),
  }),
  buildEval({
    id: 'single-math-linear',
    domain: 'Reasoning',
    prompt: 'After a 20% discount, a SaaS seat costs $48/month. What was the list price before the discount? Show your reasoning.',
    reference: 'The $48 is 80% of the list price, so list = 48 / 0.8 = $60/month.',
    output:
      'The seat is discounted by 20%, so to undo it I add 20% back:\n$48 + (20% × $48) = $48 + $9.60 = $57.60/month.',
    accuracy: d(
      3,
      'Wrong inverse operation: $48 is 80% of the list price, so list = 48 / 0.8 = $60, not $57.60. Adding 20% back to the discounted figure under-recovers because the percentage base shifted — a classic and confidently-made error.',
    ),
    clarity: d(
      9,
      'The reasoning is laid out cleanly, one step at a time — which is exactly what makes the wrong method look trustworthy.',
    ),
    completeness: d(
      8,
      'Shows the full working and commits to a single numeric answer, so the response is complete even though the result is incorrect.',
    ),
  }),
  buildEval({
    id: 'single-summary-photosynthesis',
    domain: 'Summarization',
    prompt:
      'Summarize the following in two sentences: "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy stored in glucose. It takes in carbon dioxide and water and releases oxygen as a by-product, forming the base of most food chains on Earth."',
    output:
      'Photosynthesis lets plants, algae, and some bacteria turn light energy into chemical energy stored as glucose, using carbon dioxide and water. It releases oxygen as a by-product and underpins most of Earth’s food chains.',
    accuracy: d(
      10,
      'Faithfully preserves every key fact from the source with no distortion or added claims.',
    ),
    clarity: d(
      9,
      'Crisp and well-phrased; reads naturally as a standalone summary.',
    ),
    completeness: d(
      9,
      'Captures inputs, output, energy conversion, and ecological role within the two-sentence limit. Compresses well without losing substance.',
    ),
  }),
  buildEval({
    id: 'single-factual-iphone',
    domain: 'Factual QA',
    prompt: 'In what year did the first iPhone launch, who was Apple’s CEO, and what was notable about it?',
    reference:
      'The first iPhone was released in 2007, with Steve Jobs as Apple’s CEO. It was notable for its multi-touch screen and combining a phone, iPod, and internet device. (The App Store did not arrive until 2008.)',
    output:
      'The first iPhone launched in 2007 under CEO Steve Jobs. It was groundbreaking for its multi-touch interface and shipped with the App Store, letting users download third-party apps from day one.',
    accuracy: d(
      5,
      'Year (2007) and CEO (Steve Jobs) are correct, but the claim that it shipped with the App Store is wrong — the App Store launched in 2008. A confident factual error drags accuracy down.',
    ),
    clarity: d(
      9,
      'Concise and well-structured; the prose is easy to read.',
    ),
    completeness: d(
      8,
      'Addresses all three parts of the question (year, CEO, notable feature), though the notable-feature claim is partly incorrect.',
    ),
  }),
  buildEval({
    id: 'single-support-late-order',
    domain: 'Customer Support',
    prompt:
      'A customer writes: "My order is 5 days late and I have heard nothing. This is unacceptable." Write an empathetic support reply.',
    output:
      "I'm really sorry your order is late and that you've been left without an update — that's genuinely frustrating and not the experience we want for you. I completely understand your concern and I want to make this right.",
    accuracy: d(
      8,
      'Tone and content are appropriate for the situation; nothing stated is incorrect or off-policy.',
    ),
    clarity: d(
      9,
      'Warm, well-phrased, and easy to read — the empathy lands naturally.',
    ),
    completeness: d(
      4,
      'Acknowledges the problem but offers no concrete next step: no tracking update, no timeline, no refund/reship option. An empathetic reply still needs a resolution path.',
    ),
  }),
  buildEval({
    id: 'single-translation-french',
    domain: 'Translation',
    prompt: "Translate into French: 'The weather is lovely today, shall we go for a walk?'",
    reference: "Il fait très beau aujourd'hui, et si nous allions nous promener ?",
    output: "Il fait beau aujourd'hui, on va se promener ?",
    accuracy: d(
      7,
      "Meaning is preserved and grammatical, but 'on va se promener?' reads as 'are we going for a walk?' rather than the softer suggestion 'shall we?'. 'Lovely' is also flattened to a plain 'beau'.",
    ),
    clarity: d(
      9,
      'Natural, fluent, conversational French that a native speaker would readily understand.',
    ),
    completeness: d(
      8,
      'Both clauses are translated; only the politeness/register nuance of the original is slightly lost.',
    ),
  }),
];

/* ========================================================================== */
/* 2. Batch evaluation set — populates the dashboard table + charts            */
/* ========================================================================== */

export const batchResults: ScoredEvaluation[] = [
  buildEval({
    id: 'batch-01',
    domain: 'Coding',
    prompt: 'Write a SQL query to find the second-highest salary from an Employee table.',
    output:
      'SELECT MAX(salary) FROM Employee WHERE salary < (SELECT MAX(salary) FROM Employee);',
    accuracy: d(9, 'Correctly returns the second-highest distinct salary; handles ties via the strict inequality.'),
    clarity: d(9, 'Compact, idiomatic, and easy to read.'),
    completeness: d(8, 'Solves the task; does not handle the NULL-when-no-second-salary case explicitly.'),
  }),
  buildEval({
    id: 'batch-02',
    domain: 'Coding',
    prompt: 'Explain what a race condition is to a junior developer.',
    output:
      'A race condition happens when two threads access shared data at the same time and the result depends on timing. For example, two threads incrementing the same counter can lose updates.',
    accuracy: d(9, 'Accurate definition with a correct, classic example.'),
    clarity: d(9, 'Pitched well for a junior audience.'),
    completeness: d(7, 'Defines and exemplifies but does not mention mitigations like locks or atomics.'),
  }),
  buildEval({
    id: 'batch-03',
    domain: 'Medical',
    prompt: 'What lifestyle changes help lower high blood pressure?',
    output:
      'Reduce sodium intake, exercise regularly, maintain a healthy weight, limit alcohol, and manage stress. Eating more fruits and vegetables (the DASH diet) also helps.',
    accuracy: d(10, 'All recommendations are evidence-based and correctly stated.'),
    clarity: d(9, 'Clear, scannable list of actions.'),
    completeness: d(9, 'Covers the major lifestyle levers; could add smoking cessation.'),
  }),
  buildEval({
    id: 'batch-04',
    domain: 'Medical',
    prompt: 'Is it safe to take ibuprofen and paracetamol together?',
    output:
      'Yes, they can usually be taken together because they work differently, but follow the dosing on each label and check with a pharmacist if you have other conditions.',
    accuracy: d(8, 'Generally correct guidance; appropriately hedged with a safety caveat.'),
    clarity: d(9, 'Direct answer followed by a sensible caution.'),
    completeness: d(6, 'Does not mention specific dose limits or who should avoid this combination.'),
  }),
  buildEval({
    id: 'batch-05',
    domain: 'Legal',
    prompt: 'What is the difference between a copyright and a trademark?',
    output:
      'Copyright protects original creative works like books, music, and code. A trademark protects brand identifiers like names, logos, and slogans that distinguish goods or services.',
    accuracy: d(9, 'Correctly distinguishes the two regimes and their subject matter.'),
    clarity: d(10, 'Clean, parallel structure that makes the contrast obvious.'),
    completeness: d(7, 'Covers the core distinction but omits duration and registration differences.'),
  }),
  buildEval({
    id: 'batch-06',
    domain: 'Legal',
    prompt: 'Can my landlord enter my apartment without notice?',
    output:
      'Usually no — most jurisdictions require reasonable advance notice except in emergencies. Check your lease and local tenancy laws for the exact notice period.',
    accuracy: d(8, 'Correct general principle with an appropriate jurisdictional caveat.'),
    clarity: d(9, 'Answers directly and points to the authoritative sources.'),
    completeness: d(7, 'Does not give typical notice windows (e.g., 24-48 hours) as illustrative detail.'),
  }),
  buildEval({
    id: 'batch-07',
    domain: 'Reasoning',
    prompt: 'A model is billed at $0.003 per 1K input tokens. A single request sends 12,000 input tokens. What is the input cost for that request?',
    output: '12,000 tokens ÷ 1,000 = 12 units, and 12 × $0.003 = $0.36 for the request.',
    accuracy: d(2, 'Off by 10×: 12 × $0.003 = $0.036, not $0.36. The setup (12 units of 1K tokens) is right, but the final product misplaces the decimal point.'),
    clarity: d(8, 'States the method and a single figure in one clean line.'),
    completeness: d(6, 'Gives an answer with working, but the working lands on the wrong value.'),
  }),
  buildEval({
    id: 'batch-08',
    domain: 'Reasoning',
    prompt: 'A classifier has 90% precision and 80% recall. What is its F1 score?',
    output: 'F1 = 2 × (precision × recall) / (precision + recall) = 2 × (0.9 × 0.8) / (0.9 + 0.8) = 1.44 / 1.7 ≈ 0.85.',
    accuracy: d(10, 'Correct formula and arithmetic: 2 × 0.72 / 1.7 = 1.44 / 1.7 ≈ 0.847, properly rounded to 0.85.'),
    clarity: d(9, 'Shows the formula, the substitution, and the result in one line.'),
    completeness: d(9, 'Gives the formula, the computation, and a rounded final value.'),
  }),
  buildEval({
    id: 'batch-09',
    domain: 'Summarization',
    prompt:
      'Summarize this changelog entry in one sentence: "v2.3 adds SSO via SAML, fixes a memory leak in the export worker, and deprecates the legacy /v1 API, which will be removed in v3."',
    output:
      'Version 2.3 introduces SAML-based SSO, fixes a memory leak in the export worker, and deprecates the legacy /v1 API ahead of its removal in v3.',
    accuracy: d(9, 'Preserves all three changes and the v3 removal timeline with no added or distorted claims.'),
    clarity: d(9, 'Reads as one clean, well-ordered sentence.'),
    completeness: d(8, 'Captures every item from the entry within the one-sentence limit.'),
  }),
  buildEval({
    id: 'batch-10',
    domain: 'Summarization',
    prompt: 'Condense this into a tweet: a quarterly report showing 12% revenue growth driven by international expansion and a new subscription tier.',
    output:
      'Q-results in: revenue up 12% 🚀 — powered by international growth and our new subscription tier. More markets, more ways to subscribe, more momentum.',
    accuracy: d(9, 'Preserves the two named drivers and the 12% figure accurately.'),
    clarity: d(8, 'Punchy and tweet-appropriate, though slightly marketing-heavy.'),
    completeness: d(8, 'Includes the headline number and both growth drivers.'),
  }),
  buildEval({
    id: 'batch-11',
    domain: 'Factual QA',
    prompt: 'Which HTTP status code indicates "Too Many Requests"?',
    output: 'The "Too Many Requests" response is returned with HTTP status code 503.',
    accuracy: d(2, 'Incorrect: "Too Many Requests" is 429; 503 is "Service Unavailable". A specific, confidently-stated, wrong code — the kind of factual error that passes a quick read.'),
    clarity: d(9, 'Direct and unambiguous, which is exactly what makes the wrong code convincing.'),
    completeness: d(6, 'Answers with a single code, but the code is wrong.'),
  }),
  buildEval({
    id: 'batch-12',
    domain: 'Factual QA',
    prompt: 'Is a standard JSON Web Token (JWT) encrypted by default?',
    output:
      'No — a standard JWT is signed and base64url-encoded, not encrypted. Anyone holding the token can read its payload, so secrets should never be placed in it unless you use the encrypted JWE variant.',
    accuracy: d(10, 'Correct, and addresses the common misconception: signing/encoding is not encryption, and the JWE caveat is accurate.'),
    clarity: d(9, 'Clear answer up front, then the precise reason and a practical caveat.'),
    completeness: d(10, 'Covers the default behavior, the security implication, and the encrypted alternative.'),
  }),
  buildEval({
    id: 'batch-13',
    domain: 'Customer Support',
    prompt: 'A customer asks how to reset their password. Write a helpful reply.',
    output:
      "Sure! Go to the login page and click 'Forgot password', enter your email, and we'll send you a reset link. The link expires in 30 minutes — if it doesn't arrive, check your spam folder or reply here and I'll help.",
    accuracy: d(9, 'Describes a standard, correct reset flow.'),
    clarity: d(10, 'Step-by-step, friendly, and easy to follow.'),
    completeness: d(9, 'Covers the happy path plus link expiry and a fallback if the email is missing.'),
  }),
  buildEval({
    id: 'batch-14',
    domain: 'Customer Support',
    prompt: 'Respond to a one-star review that says the app keeps crashing on startup.',
    output:
      "We're sorry the app is crashing — that's frustrating. Could you reply with your device model and app version so we can investigate? A reinstall often clears startup crashes in the meantime, and we'll prioritise a fix.",
    accuracy: d(8, 'Sensible, on-policy response; reinstall advice is reasonable.'),
    clarity: d(9, 'Empathetic, concrete, and well-structured.'),
    completeness: d(8, 'Acknowledges, gathers diagnostics, offers a workaround, and commits to a fix.'),
  }),
  buildEval({
    id: 'batch-15',
    domain: 'Translation',
    prompt: "Translate into Spanish: 'Where is the nearest train station?'",
    output: '¿Dónde está la estación de tren más cercana?',
    accuracy: d(10, 'Accurate, idiomatic translation.'),
    clarity: d(10, 'Natural phrasing with correct punctuation.'),
    completeness: d(10, 'Full, faithful translation of the request.'),
  }),
];

/* ========================================================================== */
/* 3. A/B comparison examples (both orderings + position-bias verdict)         */
/* ========================================================================== */

function buildComparison(
  base: Pick<ABComparison, 'id' | 'domain' | 'prompt' | 'outputA' | 'outputB'>,
  orderingAB: OrderingResult,
  orderingBA: OrderingResult,
): ABComparison {
  const positionBias = orderingAB.overall.winner !== orderingBA.overall.winner;
  return {
    ...base,
    orderingAB,
    orderingBA,
    positionBias,
    consistentWinner: positionBias ? null : orderingAB.overall.winner,
    model_used: MODEL,
  };
}

const v = (winner: 'A' | 'B' | 'tie', reasoning: string) => ({ winner, reasoning });

export const comparisonExamples: ABComparison[] = [
  // --- Consistent winner: clean signal, no position bias ------------------
  buildComparison(
    {
      id: 'compare-coding',
      domain: 'Coding',
      prompt: 'Write a function to reverse a linked list. Explain briefly.',
      outputA:
        'def reverse_list(head):\n    prev = None\n    while head:\n        head.next, prev, head = prev, head, head.next\n    return prev\n\nIterative: walk the list once, flipping each node’s pointer. O(n) time, O(1) space.',
      outputB:
        'You can reverse a linked list by putting all the nodes in an array, then reversing the array, then rebuilding the list from the reversed array. This works for any linked list.',
    },
    {
      firstShown: 'A',
      accuracy: v('A', 'A is a correct in-place reversal; B works but is described loosely and rebuilds the list rather than reversing pointers.'),
      clarity: v('A', 'A pairs tight code with a one-line explanation; B is prose-only with no implementation.'),
      completeness: v('A', 'A states time/space complexity; B omits any complexity analysis.'),
      overall: v('A', 'A is the stronger answer: correct, idiomatic, analyzed, and concise.'),
    },
    {
      firstShown: 'B',
      accuracy: v('A', 'Even shown second, A’s pointer-flipping reversal is the more correct and efficient solution.'),
      clarity: v('A', 'A remains clearer thanks to the concrete, well-explained code.'),
      completeness: v('A', 'A still wins on completeness with its complexity note.'),
      overall: v('A', 'Order did not change the verdict — A is consistently judged better.'),
    },
  ),
  // --- Position bias: order flips the overall winner ----------------------
  buildComparison(
    {
      id: 'compare-summary',
      domain: 'Summarization',
      prompt: 'Summarize this review in one sentence: "The laptop is fast and the screen is gorgeous, but battery life is disappointing and it runs hot under load."',
      outputA:
        'A fast laptop with a gorgeous screen, let down by disappointing battery life and heat under load.',
      outputB:
        'This laptop impresses with speed and a beautiful display, though weak battery life and thermal issues hold it back.',
    },
    {
      firstShown: 'A',
      accuracy: v('tie', 'Both faithfully capture all four points (speed, screen, battery, heat) with no distortion.'),
      clarity: v('A', 'A is marginally tighter and more direct.'),
      completeness: v('tie', 'Both include every salient point from the review.'),
      overall: v('A', 'Near-tie; A edges it on concision when shown first.'),
    },
    {
      firstShown: 'B',
      accuracy: v('tie', 'Again both are fully accurate summaries.'),
      clarity: v('B', 'Shown first, B’s slightly richer phrasing now reads as the more polished option.'),
      completeness: v('tie', 'Both remain complete.'),
      overall: v('B', 'The verdict flipped to B purely from presentation order — a textbook position-bias case on a genuine near-tie.'),
    },
  ),
  // --- Consistent winner: B is clearly better ----------------------------
  buildComparison(
    {
      id: 'compare-support',
      domain: 'Customer Support',
      prompt: 'A customer was double-charged. Write a reply.',
      outputA:
        'Sorry for the inconvenience. Please contact our billing department and they will look into the double charge for you.',
      outputB:
        "I'm sorry for the double charge — that's our mistake. I've flagged the duplicate for an immediate refund (3-5 business days) and you'll get an email confirmation. Let me know if anything else looks off.",
    },
    {
      firstShown: 'A',
      accuracy: v('B', 'B takes ownership and states a concrete remedy; A merely deflects to another department.'),
      clarity: v('B', 'B is specific and reassuring; A is vague.'),
      completeness: v('B', 'B resolves the issue (refund, timeline, confirmation); A provides no resolution.'),
      overall: v('B', 'B is decisively better — it actually fixes the customer’s problem.'),
    },
    {
      firstShown: 'B',
      accuracy: v('B', 'B’s ownership and concrete refund remain the stronger response regardless of order.'),
      clarity: v('B', 'B is still the clearer, more reassuring reply.'),
      completeness: v('B', 'B again provides the full resolution path A lacks.'),
      overall: v('B', 'Consistent across both orders — B wins on substance, not position.'),
    },
  ),
  // --- Subtle position bias: A vs tie disagreement -----------------------
  buildComparison(
    {
      id: 'compare-explanation',
      domain: 'Education',
      prompt: 'Explain recursion to a beginner in two or three sentences.',
      outputA:
        'Recursion is when a function solves a problem by calling itself on a smaller piece of the same problem, until it reaches a simple base case it can answer directly. Think of Russian nesting dolls: each one contains a smaller version until the tiniest doll, which holds nothing.',
      outputB:
        'Recursion is when a function calls itself to break a problem into smaller subproblems, stopping at a base case. It’s a common technique for things like traversing trees.',
    },
    {
      firstShown: 'A',
      accuracy: v('tie', 'Both correctly describe self-calls and the base case.'),
      clarity: v('A', 'A’s nesting-doll analogy makes the concept more vivid for a beginner.'),
      completeness: v('A', 'A’s analogy adds intuition; B is correct but drier.'),
      overall: v('A', 'A is judged better here, mainly for the beginner-friendly analogy.'),
    },
    {
      firstShown: 'B',
      accuracy: v('tie', 'Both remain accurate.'),
      clarity: v('tie', 'Shown first, B’s brevity reads as adequately clear, narrowing the gap.'),
      completeness: v('tie', 'The judge now treats B’s example and A’s analogy as roughly equivalent.'),
      overall: v('tie', 'The verdict softened from "A wins" to "tie" when the order changed — a milder but real position-bias signal.'),
    },
  ),
  // --- Consistent winner: A is correct, B hand-waves --------------------------
  buildComparison(
    {
      id: 'compare-reasoning-latency',
      domain: 'Reasoning',
      prompt:
        'A cache has a 95% hit rate. A hit costs 1ms and a miss costs 100ms. What is the average latency per request?',
      outputA:
        'Weighted average: 0.95 × 1ms + 0.05 × 100ms = 0.95 + 5 = 5.95ms per request.',
      outputB:
        'Since almost every request is a cache hit, the average latency is roughly 1ms — close to the hit cost.',
    },
    {
      firstShown: 'A',
      accuracy: v('A', 'A computes the correct expected value (5.95ms); B hand-waves to ~1ms and ignores the 5% of misses that dominate the average.'),
      clarity: v('A', 'A shows the weighted-average calculation explicitly; B asserts a number with no working.'),
      completeness: v('A', 'A accounts for both branches and their weights; B omits the miss cost entirely.'),
      overall: v('A', 'A is correct and shows its reasoning; B is intuitive but wrong.'),
    },
    {
      firstShown: 'B',
      accuracy: v('A', 'Order aside, A’s 5.95ms is the right expected value; B understates it by roughly 6×.'),
      clarity: v('A', 'A’s explicit computation stays clearer than B’s assertion.'),
      completeness: v('A', 'A still covers both branches; B still drops the miss contribution.'),
      overall: v('A', 'Consistent across both orders — A wins on correctness, not position.'),
    },
  ),
  // --- Consistent winner: A is precise, B conflates concepts ------------------
  buildComparison(
    {
      id: 'compare-factual-idempotent',
      domain: 'Factual QA',
      prompt: 'What does it mean for an HTTP method to be idempotent?',
      outputA:
        'An idempotent method leaves the server in the same state no matter how many times an identical request is sent. PUT and DELETE are idempotent; POST generally is not. Idempotent is not the same as "safe" — a safe method makes no changes at all.',
      outputB:
        'Idempotent means the method is safe and never changes anything on the server, like GET.',
    },
    {
      firstShown: 'A',
      accuracy: v('A', 'A is correct and draws the key distinction (idempotent ≠ safe); B conflates the two and wrongly implies idempotent methods never modify state.'),
      clarity: v('A', 'A is precise with correct examples; B is concise but built on a wrong premise.'),
      completeness: v('A', 'A covers the definition, examples, and the safe-vs-idempotent nuance; B misses all of it.'),
      overall: v('A', 'A is the accurate, complete answer.'),
    },
    {
      firstShown: 'B',
      accuracy: v('A', 'Shown first or not, B’s "idempotent means safe" is incorrect; A remains right.'),
      clarity: v('A', 'A’s precision wins regardless of order.'),
      completeness: v('A', 'A still covers the nuance B omits.'),
      overall: v('A', 'Consistent both ways — A wins on substance.'),
    },
  ),
  // --- Consistent winner: A keeps the causal chain, B is vague ----------------
  buildComparison(
    {
      id: 'compare-summary-incident',
      domain: 'Summarization',
      prompt:
        'Summarize in one sentence: "The deploy was rolled back because a migration locked the orders table for 40s, tripping the health check, which the load balancer treated as an outage."',
      outputA:
        'The deploy was rolled back after a migration locked the orders table for 40s, failing the health check and signaling an outage to the load balancer.',
      outputB: 'The deployment hit a database problem and was automatically rolled back.',
    },
    {
      firstShown: 'A',
      accuracy: v('A', 'A preserves the causal chain (lock → health-check failure → load-balancer rollback); B is accurate but too vague to convey the cause.'),
      clarity: v('A', 'A is specific yet readable; B is clear but uninformative.'),
      completeness: v('A', 'A keeps every link in the chain; B collapses it to "a database problem".'),
      overall: v('A', 'A is the faithful, informative summary.'),
    },
    {
      firstShown: 'B',
      accuracy: v('A', 'A’s causal detail beats B’s vagueness in either order.'),
      clarity: v('A', 'A stays the more informative summary.'),
      completeness: v('A', 'A retains the chain B drops.'),
      overall: v('A', 'Consistent both ways — A wins on faithfulness.'),
    },
  ),
];

/* ========================================================================== */
/* 4. Aggregate statistics for the dashboard                                   */
/* ========================================================================== */

function mean(nums: number[]): number {
  if (nums.length === 0) return 0;
  return Math.round((nums.reduce((s, n) => s + n, 0) / nums.length) * 100) / 100;
}

export function getDashboardStats(rows: ScoredEvaluation[] = batchResults): DashboardStats {
  const domains = new Set(rows.map((r) => r.domain));
  const biased = comparisonExamples.filter((c) => c.positionBias).length;

  return {
    totalEvaluations: rows.length,
    averageTotal: mean(rows.map((r) => r.total_score)),
    averageAccuracy: mean(rows.map((r) => r.accuracy.score)),
    averageClarity: mean(rows.map((r) => r.clarity.score)),
    averageCompleteness: mean(rows.map((r) => r.completeness.score)),
    domainCount: domains.size,
    comparisonCount: comparisonExamples.length,
    positionBiasRate:
      comparisonExamples.length === 0
        ? 0
        : Math.round((biased / comparisonExamples.length) * 100),
  };
}

/** Per-domain average total score, for the dashboard's domain chart. */
export function getDomainAverages(rows: ScoredEvaluation[] = batchResults) {
  const byDomain = new Map<string, number[]>();
  for (const r of rows) {
    const list = byDomain.get(r.domain) ?? [];
    list.push(r.total_score);
    byDomain.set(r.domain, list);
  }
  return Array.from(byDomain.entries())
    .map(([domain, scores]) => ({ domain, average: mean(scores), count: scores.length }))
    .sort((a, b) => b.average - a.average);
}

/** Histogram of total scores bucketed by integer band, for the dashboard. */
export function getScoreDistribution(rows: ScoredEvaluation[] = batchResults) {
  const bands = [
    { band: '0-2', min: 0, max: 2 },
    { band: '2-4', min: 2, max: 4 },
    { band: '4-6', min: 4, max: 6 },
    { band: '6-8', min: 6, max: 8 },
    { band: '8-10', min: 8, max: 10.01 },
  ];
  return bands.map(({ band, min, max }) => ({
    band,
    count: rows.filter((r) => r.total_score >= min && r.total_score < max).length,
  }));
}

/** Average per criterion, for the dashboard's criterion chart. */
export function getCriterionAverages(rows: ScoredEvaluation[] = batchResults) {
  return [
    { criterion: 'Accuracy', average: mean(rows.map((r) => r.accuracy.score)) },
    { criterion: 'Clarity', average: mean(rows.map((r) => r.clarity.score)) },
    { criterion: 'Completeness', average: mean(rows.map((r) => r.completeness.score)) },
  ];
}
