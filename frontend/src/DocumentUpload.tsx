import { useRef, useState } from 'react';
import type { FormEvent } from 'react';

type UploadedDocument = {
  id: string;
  filename: string;
  page_count: number;
  chunk_count: number;
  embedding_model: string;
  embedding_dimension: number;
  embedding_revision: string;
  embedding_profile: string;
};

function isDocument(value: unknown): value is UploadedDocument {
  if (typeof value !== 'object' || value === null) return false;
  const data = value as Record<string, unknown>;
  return typeof data.id === 'string' && typeof data.filename === 'string'
    && typeof data.page_count === 'number' && typeof data.chunk_count === 'number'
    && typeof data.embedding_model === 'string' && typeof data.embedding_dimension === 'number'
    && typeof data.embedding_revision === 'string' && typeof data.embedding_profile === 'string';
}

export function DocumentUpload() {
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<UploadedDocument | null>(null);

  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const file = input.current?.files?.[0];
    setError(''); setResult(null);
    if (!file) { setError('Select a PDF first.'); return; }
    if (!file.name.toLowerCase().endsWith('.pdf') || (file.type && file.type !== 'application/pdf')) {
      setError('Please select a PDF file.'); return;
    }
    setBusy(true);
    try {
      const body = new FormData();
      body.append('file', file);
      const response = await fetch('/api/documents', { method: 'POST', body });
      const data: unknown = await response.json();
      if (!response.ok) {
        let message = `Upload failed (HTTP ${response.status}).`;
        if (typeof data === 'object' && data !== null && 'detail' in data) {
          const detail = data.detail;
          if (typeof detail === 'string') message = detail;
          else if (typeof detail === 'object' && detail !== null && 'message' in detail && typeof detail.message === 'string') message = detail.message;
        }
        throw new Error(message);
      }
      if (!isDocument(data)) throw new Error('The server returned an unexpected response.');
      setResult(data);
      if (input.current) input.current.value = '';
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Upload failed. Please try again.');
    } finally { setBusy(false); }
  }

  return <section className="upload-panel" aria-labelledby="upload-title" aria-busy={busy}>
    <h2 id="upload-title">Document ingestion</h2>
    <p>Upload a text-based PDF to extract pages, create overlapping chunks and store local embedding vectors. Scans and encrypted PDFs are unsupported. Default limit: 10 MiB; server limits apply.</p>
    <form onSubmit={upload}>
      <label htmlFor="pdf-file">PDF document</label>
      <input ref={input} id="pdf-file" type="file" accept=".pdf,application/pdf" disabled={busy} onChange={() => { setError(''); setResult(null); }} />
      <button type="submit" disabled={busy}>{busy ? 'Extracting and storing...' : 'Upload PDF'}</button>
    </form>
    <div role="status" aria-live="polite">{busy && <p>Processing your PDF. This may take a moment.</p>}{result && <div className="upload-success"><strong>{result.filename} was stored.</strong><p>{result.page_count} pages / {result.chunk_count} chunks / {result.embedding_dimension} dimensions</p><p>Document ID: <code>{result.id}</code></p><p>Model: {result.embedding_model} @ {result.embedding_revision.slice(0, 12)}. Retrieval and question answering are not implemented.</p></div>}</div>
    {error && <p className="upload-error" role="alert">{error}</p>}
  </section>;
}
