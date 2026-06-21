/**
 * Batch CSV evaluation. Stateless: the CSV is parsed in the browser, scored by
 * the live judge with bounded concurrency, and rendered in-session. Nothing is
 * uploaded or stored.
 */

import UploadForm from '@/components/UploadForm';

export default function BatchPage() {
  return (
    <div className="space-y-6">
      <div>
        <span className="kicker">
          <span className="h-1.5 w-1.5 rounded-full bg-indigo-500" />
          Bulk · concurrent
        </span>
        <h1 className="mt-2 text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">Batch evaluation from CSV</h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-600">
          Score many outputs at once. Your file is parsed locally and the rows are judged concurrently — no database,
          no upload. Results stay in this session.
        </p>
      </div>
      <UploadForm />
    </div>
  );
}
