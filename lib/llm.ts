/**
 * LLM client for evaluation using FastRouter (OpenAI-compatible API)
 * Uses the LLM-as-judge pattern to score outputs against expected results
 */

import OpenAI from 'openai';
import { RubricScores } from '@/types';

// Use placeholder during build if env var is not set
const apiKey = process.env.FASTROUTER_API_KEY || 'placeholder-api-key';

const client = new OpenAI({
  apiKey,
  baseURL: 'https://api.fastrouter.ai/api/v1',
});

/**
 * The model used for evaluation (configurable via env var)
 */
const MODEL_NAME = process.env.LLM_MODEL || 'openai/gpt-4o-mini';

/**
 * Generate actual output using the LLM based on a prompt
 * This simulates what an AI would respond to the given prompt
 */
export async function generateOutput(prompt: string): Promise<string> {
  try {
    const response = await client.chat.completions.create({
      model: MODEL_NAME,
      messages: [{ role: 'user', content: prompt }],
    });

    const text = response.choices[0]?.message?.content;
    if (!text) {
      throw new Error('Empty response from LLM');
    }

    return text;
  } catch (error: any) {
    throw new Error(`Failed to generate output: ${error.message}`);
  }
}

/**
 * Evaluate an actual output against an expected output using LLM-as-judge methodology
 *
 * @param prompt - The original prompt given to the AI
 * @param expectedOutput - The expected/ideal response
 * @param actualOutput - The actual response to evaluate
 * @returns RubricScores - Scores for accuracy, clarity, and completeness (0-10 each)
 */
export async function evaluateWithLLM(
  prompt: string,
  expectedOutput: string,
  actualOutput: string
): Promise<RubricScores> {
  try {
    const evaluationPrompt = `You are an expert evaluator. Given a prompt, expected output, and actual output, score the actual output on three criteria:

1. **Accuracy (0-10)**: How factually correct is the actual output compared to the expected output? Does it contain the same key information and facts?
   - 0-3: Mostly incorrect or contradicts expected output
   - 4-6: Partially correct, some key points are accurate
   - 7-8: Mostly correct with minor inaccuracies
   - 9-10: Fully accurate, matches expected output

2. **Clarity (0-10)**: How clear, understandable, and well-structured is the actual output?
   - 0-3: Confusing, poorly structured, hard to understand
   - 4-6: Somewhat clear but could be better organized
   - 7-8: Clear and well-structured
   - 9-10: Exceptionally clear, concise, and well-organized

3. **Completeness (0-10)**: Does the actual output cover all the important points from the expected output?
   - 0-3: Missing most key points
   - 4-6: Covers some key points but misses important ones
   - 7-8: Covers most key points
   - 9-10: Comprehensive, covers all key points

**Prompt:**
${prompt}

**Expected Output:**
${expectedOutput}

**Actual Output:**
${actualOutput}

Respond ONLY with valid JSON in this exact format (no other text, no markdown, no explanation):
{"accuracy": X, "clarity": Y, "completeness": Z}

Where X, Y, and Z are integers from 0 to 10.`;

    const response = await client.chat.completions.create({
      model: MODEL_NAME,
      messages: [{ role: 'user', content: evaluationPrompt }],
    });

    const text = response.choices[0]?.message?.content;
    if (!text) {
      throw new Error('Empty response from LLM');
    }

    // Parse the JSON response
    // Remove any markdown code blocks if present
    let jsonText = text.trim();
    if (jsonText.startsWith('```json')) {
      jsonText = jsonText.replace(/```json\n?/, '').replace(/\n?```$/, '');
    } else if (jsonText.startsWith('```')) {
      jsonText = jsonText.replace(/```\n?/, '').replace(/\n?```$/, '');
    }

    const scores: RubricScores = JSON.parse(jsonText);

    // Validate scores are within range
    if (
      typeof scores.accuracy !== 'number' ||
      typeof scores.clarity !== 'number' ||
      typeof scores.completeness !== 'number' ||
      scores.accuracy < 0 || scores.accuracy > 10 ||
      scores.clarity < 0 || scores.clarity > 10 ||
      scores.completeness < 0 || scores.completeness > 10
    ) {
      throw new Error('Invalid scores returned from LLM');
    }

    return scores;
  } catch (error: any) {
    console.error('Error evaluating with LLM:', error);
    throw new Error(`Failed to evaluate with LLM: ${error.message}`);
  }
}

/**
 * Calculate the total score from rubric scores (average of all three)
 */
export function calculateTotalScore(scores: RubricScores): number {
  const total = (scores.accuracy + scores.clarity + scores.completeness) / 3;
  return Math.round(total * 10) / 10; // Round to 1 decimal place
}
