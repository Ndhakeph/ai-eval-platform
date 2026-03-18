/**
 * Core evaluation logic
 * Orchestrates the evaluation process from fetching test cases to storing results
 */

import { getTestCaseById, insertEvaluationResult } from './supabase';
import { evaluateWithLLM, calculateTotalScore, generateOutput } from './llm';
import { EvaluationResult } from '@/types';

/**
 * Run an evaluation for a test case
 *
 * @param testCaseId - The ID of the test case to evaluate
 * @param actualOutput - The actual output to evaluate (if provided)
 *                       If not provided, will generate output using the LLM
 * @returns Promise<EvaluationResult> - The evaluation result
 */
export async function runEvaluation(
  testCaseId: string,
  actualOutput?: string
): Promise<EvaluationResult> {
  try {
    // 1. Fetch the test case from Supabase
    const testCase = await getTestCaseById(testCaseId);

    if (!testCase) {
      throw new Error(`Test case with ID ${testCaseId} not found`);
    }

    // 2. Generate actual output if not provided
    let outputToEvaluate = actualOutput;
    if (!outputToEvaluate) {
      console.log('Generating output for prompt:', testCase.prompt);
      outputToEvaluate = await generateOutput(testCase.prompt);
    }

    // 3. Evaluate the output using LLM-as-judge
    console.log('Evaluating output against expected output...');
    const scores = await evaluateWithLLM(
      testCase.prompt,
      testCase.expected_output,
      outputToEvaluate
    );

    // 4. Calculate total score
    const totalScore = calculateTotalScore(scores);

    // 5. Store the result in Supabase
    const result = await insertEvaluationResult({
      test_case_id: testCaseId,
      actual_output: outputToEvaluate,
      accuracy_score: scores.accuracy,
      clarity_score: scores.clarity,
      completeness_score: scores.completeness,
      total_score: totalScore,
      model_used: process.env.LLM_MODEL || 'openai/gpt-4o-mini',
    });

    // 6. Return the formatted result
    return {
      id: result.id,
      test_case_id: testCaseId,
      actual_output: outputToEvaluate,
      scores: {
        accuracy: scores.accuracy,
        clarity: scores.clarity,
        completeness: scores.completeness,
      },
      total_score: totalScore,
      model_used: process.env.LLM_MODEL || 'openai/gpt-4o-mini',
      created_at: result.created_at,
    };
  } catch (error: any) {
    console.error('Error running evaluation:', error);
    throw new Error(`Failed to run evaluation: ${error.message}`);
  }
}

/**
 * Run evaluations for multiple test cases
 * Processes them sequentially to avoid rate limiting
 *
 * @param testCaseIds - Array of test case IDs to evaluate
 * @param onProgress - Optional callback for progress updates
 * @returns Promise<EvaluationResult[]> - Array of evaluation results
 */
export async function runBatchEvaluation(
  testCaseIds: string[],
  onProgress?: (completed: number, total: number) => void
): Promise<EvaluationResult[]> {
  const results: EvaluationResult[] = [];
  const total = testCaseIds.length;

  for (let i = 0; i < testCaseIds.length; i++) {
    try {
      const result = await runEvaluation(testCaseIds[i]);
      results.push(result);

      if (onProgress) {
        onProgress(i + 1, total);
      }

      // Add a small delay to avoid rate limiting (1 second between requests)
      if (i < testCaseIds.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    } catch (error: any) {
      console.error(`Failed to evaluate test case ${testCaseIds[i]}:`, error);
      // Continue with next test case instead of failing entirely
    }
  }

  return results;
}
