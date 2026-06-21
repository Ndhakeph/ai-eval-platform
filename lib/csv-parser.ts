/**
 * Client-side CSV parsing for batch evaluation.
 *
 * Parsing happens entirely in the browser (Papa Parse) — nothing is uploaded or
 * stored. The parsed rows are sent to the live judge endpoint and scored in
 * memory.
 *
 * Expected columns: `prompt`, `output`, and an optional `reference` (the gold
 * answer to judge against). Header names are normalized, and a couple of common
 * aliases are accepted so realistic files just work.
 */

import Papa from 'papaparse';
import { TestCaseCSVRow } from '@/types';

const MAX_SIZE = 5 * 1024 * 1024; // 5MB

/** Map common header aliases to our canonical column names. */
function canonicalHeader(header: string): string {
  const h = header.trim().toLowerCase().replace(/\s+/g, '_');
  if (h === 'actual_output' || h === 'response' || h === 'answer' || h === 'completion') return 'output';
  if (h === 'expected_output' || h === 'expected' || h === 'gold' || h === 'reference_answer') return 'reference';
  return h;
}

export function validateCSVFile(file: File): { valid: boolean; error?: string } {
  if (!file.name.toLowerCase().endsWith('.csv')) {
    return { valid: false, error: 'Invalid file type. Please upload a .csv file.' };
  }
  if (file.size > MAX_SIZE) {
    return { valid: false, error: 'File too large. Maximum size is 5MB.' };
  }
  return { valid: true };
}

export async function parseTestCases(file: File): Promise<TestCaseCSVRow[]> {
  return new Promise((resolve, reject) => {
    const validation = validateCSVFile(file);
    if (!validation.valid) {
      reject(new Error(validation.error));
      return;
    }

    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: true,
      transformHeader: canonicalHeader,
      complete: (results) => {
        try {
          if (!results.data || results.data.length === 0) {
            reject(new Error('CSV file is empty or contains no valid rows.'));
            return;
          }

          const rows: TestCaseCSVRow[] = [];
          const errors: string[] = [];

          results.data.forEach((row, index) => {
            const rowNumber = index + 2; // +1 for header, +1 for 1-based display
            const prompt = (row.prompt ?? '').trim();
            const output = (row.output ?? '').trim();
            const reference = (row.reference ?? '').trim();

            if (!prompt) {
              errors.push(`Row ${rowNumber}: missing 'prompt'`);
              return;
            }
            if (!output) {
              errors.push(`Row ${rowNumber}: missing 'output'`);
              return;
            }

            rows.push({ prompt, output, reference: reference || undefined });
          });

          if (rows.length === 0) {
            reject(
              new Error(
                `No valid rows found. Ensure your CSV has 'prompt' and 'output' columns.\n${errors.join('\n')}`,
              ),
            );
            return;
          }

          resolve(rows);
        } catch (error) {
          reject(new Error(`Failed to process CSV: ${(error as Error).message}`));
        }
      },
      error: (error) => reject(new Error(`Failed to parse CSV: ${error.message}`)),
    });
  });
}

/** A downloadable starter template demonstrating the expected columns. */
export function generateSampleCSV(): string {
  return Papa.unparse([
    {
      prompt: 'What is the capital of France?',
      output: 'The capital of France is Paris.',
      reference: 'Paris',
    },
    {
      prompt: 'Write a haiku about the ocean.',
      output: 'Endless blue expanse / waves whisper to the shoreline / salt air fills my lungs',
      reference: '',
    },
    {
      prompt: 'Explain HTTP status code 404 in one sentence.',
      output: 'A 404 means the server could not find the requested resource.',
      reference: '404 Not Found indicates the requested resource does not exist on the server.',
    },
  ]);
}
