import { useEffect, useRef, useState } from 'react';
import type { FormEvent, ReactNode } from 'react';

type DocumentItem = {
  id: string;
  filename: string;
  page_count: number;
  chunk_count: number;
};
type DocumentPage = { documents: DocumentItem[]; total: number; limit: number; offset: number };
type RagConfig = {
  default_top_k: number;
  max_top_k: number;
  min_similarity: number;
  llm_provider: string;
  llm_model: string;
  max_question_chars: number;
};
type Citation = {
  source_id: string;
  chunk_id: number;
  document_id: string;
  filename: string;
  page_number: number;
  chunk_order: number;
  quote: string;
  text: string;
};
type Usage = { prompt_tokens: number | null; completion_tokens: number | null };
type RagResult = {
  status: 'answered' | 'insufficient_evidence';
  answer: string | null;
  message: string | null;
  citations: Citation[];
  retrieval_ms: number;
  generation_ms: number;
  retrieved_chunk_count: number;
  context_chunk_count: number;
  embedding_profile: string;
  llm_provider: string;
  llm_model: string;
  usage: Usage | null;
};

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}
function count(value: unknown, minimum = 0): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= minimum;
}
function duration(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0;
}
function isDocument(value: unknown): value is DocumentItem {
  return record(value) && typeof value.id === 'string' && typeof value.filename === 'string'
    && count(value.page_count, 1) && count(value.chunk_count, 1);
}
function isPage(value: unknown): value is DocumentPage {
  return record(value) && Array.isArray(value.documents) && value.documents.every(isDocument)
    && count(value.total) && count(value.limit, 1) && count(value.offset);
}
function isConfig(value: unknown): value is RagConfig {
  return record(value) && count(value.default_top_k, 1) && count(value.max_top_k, 1)
    && value.default_top_k <= value.max_top_k && count(value.max_question_chars, 1)
    && typeof value.min_similarity === 'number' && Number.isFinite(value.min_similarity)
    && typeof value.llm_provider === 'string' && typeof value.llm_model === 'string';
}
function isCitation(value: unknown): value is Citation {
  return record(value) && typeof value.source_id === 'string' && /^S[1-9]\d*$/.test(value.source_id)
    && count(value.chunk_id, 1) && typeof value.document_id === 'string'
    && typeof value.filename === 'string' && count(value.page_number, 1) && count(value.chunk_order)
    && typeof value.quote === 'string' && typeof value.text === 'string';
}
function isUsage(value: unknown): value is Usage {
  return record(value) && (value.prompt_tokens === null || count(value.prompt_tokens))
    && (value.completion_tokens === null || count(value.completion_tokens));
}
function isResult(value: unknown): value is RagResult {
  if (!record(value) || !Array.isArray(value.citations) || !value.citations.every(isCitation)) return false;
  if (new Set(value.citations.map(item => item.source_id)).size !== value.citations.length) return false;
  const validAnswer = value.status === 'answered' && typeof value.answer === 'string' && value.answer.trim().length > 0
    && value.citations.length > 0 && (value.message === null || typeof value.message === 'string');
  const validMissing = value.status === 'insufficient_evidence' && value.answer === null
    && typeof value.message === 'string' && value.citations.length === 0;
  return (validAnswer || validMissing) && duration(value.retrieval_ms) && duration(value.generation_ms)
    && count(value.retrieved_chunk_count) && count(value.context_chunk_count)
    && typeof value.embedding_profile === 'string' && typeof value.llm_provider === 'string'
    && typeof value.llm_model === 'string' && (value.usage === null || isUsage(value.usage));
}

async function requestJson(url: string, options: RequestInit = {}): Promise<unknown> {
  const response = await fetch(url, options);
  let body: unknown;
  try { body = await response.json(); }
  catch { throw new Error(`The API returned an unreadable response (HTTP ${response.status}).`); }
  if (!response.ok) {
    let message = `Request failed (HTTP ${response.status}).`;
    if (record(body)) {
      const detail = body.detail;
      if (typeof detail === 'string') message = detail;
      else if (record(detail) && typeof detail.message === 'string') message = detail.message;
      else if (Array.isArray(detail)) {
        const messages = detail.filter(record).map(item => item.msg).filter((item): item is string => typeof item === 'string');
        if (messages.length) message = messages.join('; ');
      }
    }
    throw new Error(message);
  }
  return body;
}

function answerWithLinks(answer: string, citations: Citation[]): ReactNode[] {
  const sources = new Set(citations.map(citation => citation.source_id));
  return answer.split(/(\[S[1-9]\d*\])/g).map((part, index) => {
    const source = part.slice(1, -1);
    return /^\[S[1-9]\d*\]$/.test(part) && sources.has(source)
      ? <a key={index} href={`#rag-source-${source}`}>{part}</a>
      : <span key={index}>{part}</span>;
  });
}

export function RagQuestion({ libraryVersion }: { libraryVersion: number }) {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [total, setTotal] = useState(0);
  const [nextOffset, setNextOffset] = useState(0);
  const [config, setConfig] = useState<RagConfig | null>(null);
  const [reload, setReload] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [libraryError, setLibraryError] = useState('');
  const [allDocuments, setAllDocuments] = useState(true);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<RagResult | null>(null);
  const answerRequest = useRef<AbortController | null>(null);
  const moreRequest = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    answerRequest.current?.abort();
    answerRequest.current = null;
    moreRequest.current?.abort();
    moreRequest.current = null;
    setBusy(false); setLoadingMore(false); setLoading(true); setLibraryError('');
    setResult(null); setError(''); setSelectedIds([]);
    Promise.all([
      requestJson('/api/documents?limit=100&offset=0', { signal: controller.signal }),
      requestJson('/api/rag/config', { signal: controller.signal }),
    ]).then(([page, settings]) => {
      if (!isPage(page) || !isConfig(settings)) throw new Error('The API returned an unexpected library/configuration response.');
      if (!controller.signal.aborted) {
        setDocuments(page.documents); setTotal(page.total);
        setNextOffset(page.offset + page.documents.length);
        setConfig(settings); setTopK(settings.default_top_k);
      }
    }).catch(reason => {
      if (!controller.signal.aborted) setLibraryError(reason instanceof Error ? reason.message : 'The document library could not be loaded.');
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => { controller.abort(); moreRequest.current?.abort(); answerRequest.current?.abort(); };
  }, [libraryVersion, reload]);

  async function loadMore() {
    if (moreRequest.current || loading || busy) return;
    const controller = new AbortController(); moreRequest.current = controller;
    setLoadingMore(true); setLibraryError('');
    try {
      const page = await requestJson(`/api/documents?limit=100&offset=${nextOffset}`, { signal: controller.signal });
      if (!isPage(page)) throw new Error('The API returned an unexpected document list.');
      if (!controller.signal.aborted) {
        setDocuments(previous => [...new Map([...previous, ...page.documents].map(item => [item.id, item])).values()]);
        setTotal(page.total); setNextOffset(page.documents.length ? page.offset + page.documents.length : page.total);
      }
    } catch (reason) {
      if (!controller.signal.aborted) setLibraryError(reason instanceof Error ? reason.message : 'More documents could not be loaded.');
    } finally {
      if (moreRequest.current === controller) { moreRequest.current = null; setLoadingMore(false); }
    }
  }

  async function ask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (answerRequest.current) return;
    setError(''); setResult(null);
    const value = question.trim();
    if (!config || loading || libraryError) { setError('Load the document library and configuration before asking a question.'); return; }
    if (!value) { setError('Enter a question first.'); return; }
    if (value.length > config.max_question_chars) { setError(`Questions must be at most ${config.max_question_chars} characters.`); return; }
    if (!Number.isInteger(topK) || topK < 1 || topK > config.max_top_k) { setError(`Choose between 1 and ${config.max_top_k} chunks.`); return; }
    if (!total) { setError('Upload a PDF before asking a question.'); return; }
    if (!allDocuments && !selectedIds.length) { setError('Select at least one document or choose all documents.'); return; }
    const controller = new AbortController(); answerRequest.current = controller; setBusy(true);
    try {
      const data = await requestJson('/api/rag/questions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
        body: JSON.stringify({ question: value, top_k: topK, document_ids: allDocuments ? null : selectedIds }),
      });
      if (!isResult(data)) throw new Error('The API returned an unexpected answer response.');
      if (!controller.signal.aborted) setResult(data);
    } catch (reason) {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'The question could not be answered.');
    } finally {
      if (answerRequest.current === controller) { answerRequest.current = null; setBusy(false); }
    }
  }

  function clearAnswer() { setResult(null); setError(''); }

  return <section className="rag-panel" aria-labelledby="rag-question-title" aria-busy={busy}>
    <div className="rag-panel-heading"><div><h2 id="rag-question-title">Ask your documents</h2><p>Answers use retrieved passages from your PDFs. Review the supporting quotes before relying on an answer.</p></div><button className="secondary-button" type="button" disabled={loading || busy} onClick={() => setReload(value => value + 1)}>Refresh library</button></div>
    {config && <p className="provider-label">Configured provider: {config.llm_provider} / {config.llm_model}. Availability is checked when answering.</p>}
    {loading && <p role="status">Loading documents and question settings...</p>}
    {libraryError && <p className="rag-error" role="alert">{libraryError} <button className="text-button" type="button" disabled={busy || loading} onClick={() => setReload(value => value + 1)}>Retry library</button></p>}
    {!loading && !libraryError && !total && <p className="rag-empty">No documents are stored yet. Upload a text-based PDF above to begin.</p>}
    <form onSubmit={ask}>
      <fieldset disabled={busy || loading || !config}>
        <legend>Evidence scope</legend>
        <label className="scope-option"><input type="checkbox" checked={allDocuments} onChange={event => { setAllDocuments(event.target.checked); clearAnswer(); }} /> All {total || ''} documents</label>
        {!allDocuments && <div className="document-picker">
          <p>Select the documents that may support this answer. No selection searches nothing.</p>
          <div className="document-options">{documents.map(document => <label key={document.id} className="document-option"><input type="checkbox" checked={selectedIds.includes(document.id)} onChange={event => { setSelectedIds(previous => event.target.checked ? [...previous, document.id] : previous.filter(id => id !== document.id)); clearAnswer(); }} /><span><strong>{document.filename}</strong><small>{document.page_count} pages / {document.chunk_count} chunks</small></span></label>)}</div>
          <p className="selection-summary">{selectedIds.length} selected / Showing {documents.length} of {total}</p>
          {nextOffset < total && <button className="secondary-button" type="button" disabled={loadingMore || busy} onClick={loadMore}>{loadingMore ? 'Loading more...' : 'Load more documents'}</button>}
        </div>}
        <label className="question-label" htmlFor="rag-question">Your question</label>
        <textarea id="rag-question" value={question} maxLength={config?.max_question_chars} rows={4} placeholder="Ask a question that your uploaded documents can answer." onChange={event => { setQuestion(event.target.value); clearAnswer(); }} />
        <div className="question-actions"><label htmlFor="rag-top-k">Maximum retrieved chunks <input id="rag-top-k" type="number" min={1} max={config?.max_top_k} value={Number.isNaN(topK) ? '' : topK} onChange={event => { setTopK(event.target.valueAsNumber); clearAnswer(); }} /></label><button type="submit" disabled={busy || loading || !config || !!libraryError || !total}>{busy ? 'Retrieving and generating...' : 'Ask question'}</button></div>
      </fieldset>
    </form>
    <div role="status" aria-live="polite">{busy && <p>Retrieving relevant passages and asking the configured model. Local generation can take a moment.</p>}</div>
    {error && <p className="rag-error" role="alert">{error}</p>}
    {result && <div className="rag-result" aria-live="polite">
      {result.status === 'insufficient_evidence' ? <div className="rag-insufficient"><h3>Insufficient evidence</h3><p>{result.message}</p></div> : <><h3>Grounded answer</h3><div className="rag-answer">{answerWithLinks(result.answer || '', result.citations)}</div><h3 className="sources-heading">Supporting sources</h3><div className="rag-citations">{result.citations.map(citation => <article key={citation.source_id} id={`rag-source-${citation.source_id}`} className="rag-citation"><h4><span>[{citation.source_id}]</span> {citation.filename} / page {citation.page_number}</h4><blockquote>{citation.quote}</blockquote><details><summary>Supporting chunk {citation.chunk_id}</summary><p className="citation-metadata">Document: <code>{citation.document_id}</code> / chunk order {citation.chunk_order}</p><pre>{citation.text}</pre></details></article>)}</div></>}
      <p className="rag-metrics">{result.retrieved_chunk_count} chunks retrieved / {result.context_chunk_count} used in context / Retrieval {result.retrieval_ms.toFixed(0)} ms / Generation {result.generation_ms.toFixed(0)} ms</p>
      <p className="rag-metrics">{result.llm_provider} / {result.llm_model}{result.usage && <> / Prompt tokens: {result.usage.prompt_tokens ?? 'unavailable'} / Output tokens: {result.usage.completion_tokens ?? 'unavailable'}</>}</p>
    </div>}
  </section>;
}
