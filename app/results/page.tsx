/**
 * Legacy route. The dashboard is now the canonical results view, so `/results`
 * permanently redirects there. (Kept as a redirect rather than deleted so old
 * links don't 404.)
 */

import { redirect } from 'next/navigation';

export default function ResultsRedirect() {
  redirect('/');
}
