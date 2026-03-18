/**
 * API route for running evaluations
 * POST /api/evaluate
 */

import { NextRequest, NextResponse } from 'next/server';
import { runEvaluation, runBatchEvaluation } from '@/lib/evaluator';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { testCaseId, testCaseIds, actualOutput } = body;

    // Single evaluation
    if (testCaseId) {
      const result = await runEvaluation(testCaseId, actualOutput);
      return NextResponse.json({
        success: true,
        result,
      });
    }

    // Batch evaluation
    if (testCaseIds && Array.isArray(testCaseIds)) {
      if (testCaseIds.length === 0) {
        return NextResponse.json(
          { error: 'testCaseIds array cannot be empty' },
          { status: 400 }
        );
      }

      const results = await runBatchEvaluation(testCaseIds);
      return NextResponse.json({
        success: true,
        results,
        totalEvaluated: results.length,
      });
    }

    return NextResponse.json(
      { error: 'Invalid request: either testCaseId or testCaseIds is required' },
      { status: 400 }
    );
  } catch (error: any) {
    console.error('Error running evaluation:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to run evaluation' },
      { status: 500 }
    );
  }
}

// Max timeout for Vercel Hobby plan (60s)
export const maxDuration = 60;
