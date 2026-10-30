import { useEffect, useRef, useState } from 'react';
import type { FormEvent, ReactNode } from 'react';

import {
  isDocumentPage, requestJson, submitComparison,
  type AgentComparison, type Citation, type ComparisonResponse, type DocumentItem,
  type RagComparison, type ScopeMode, type Strategy,
} from './CompareContract.js';

const strategies: { value: Strategy; label: string }[] = [
  { value: 'semantic', label: 'Semantic' },
  { value: 'hybrid', label: 'Hybrid (semantic + keyword)' },
  { value: 'hybrid_rerank', label: 'Hybrid + cross-encoder reranking' },
];

function formatDuration(milliseconds: number | null): string {
  if (milliseconds === null) return 'Unavailable';
  return milliseconds >= 1000
    ? `${(milliseconds / 1000).toFixed(2)} s`
    : `${milliseconds.toFixed(0)} ms`;
}

function label(value: string | null): string {
  return value ? value.replaceAll('_', ' ') : 'Unavailable';
}

function answerWithLinks(answer: string, citations: Citation[], pipeline: 'rag' | 'agentic'): ReactNode[] {
  const sources = new Set(citations.map(item => item.source_id));
  return answer.split(/(\[S[1-9]\d*\])/g).map((part, index) => {
    const source = part.slice(1, -1);
    return /^\[S[1-9]\d*\]$/.test(part) && sources.has(source)
      ? <a key={index} href={`#compare-${pipeline}-source-${source}`}>{part}</a>
      : <span key={index}>{part}</span>;
  });
}

function CitationList({ citations, pipeline }: { citations: Citation[]; pipeline: 'rag' | 'agentic' }) {
  if (!citations.length) return null;
  return <div className="comparison-citations">
    <h4>Validated citations</h4>
    {citations.map(item => <article key={item.source_id} id={`compare-${pipeline}-source-${item.source_id}`}>
      <h5><span>[{item.source_id}]</span> {item.filename} / page {item.page_number}</h5>
      <blockquote>{item.quote}</blockquote>
      <p>Stored chunk {item.chunk_id} / order {item.chunk_order}</p>
    </article>)}
  </div>;
}

function ErrorOutcome({ result }: { result: RagComparison | AgentComparison }) {
  if (!result.error) return null;
  return <div className="comparison-error-outcome" role="status">
    <h4>Pipeline error</h4>
    <p>{result.error.message}</p>
    <p><code>{result.error.code}</code> / {result.error.category} / {result.error.retryable ? 'retryable' : 'not retryable'}</p>
  </div>;
}

function UsageMetrics({ prompt, completion }: { prompt: number | null; completion: number | null }) {
  return <>
    <div><dt>Prompt tokens</dt><dd>{prompt ?? 'Unavailable'}</dd></div>
    <div><dt>Output tokens</dt><dd>{completion ?? 'Unavailable'}</dd></div>
  </>;
}

function RagResultCard({ result }: { result: RagComparison }) {
  return <article className={`comparison-result-card outcome-${result.status}`} aria-labelledby="comparison-rag-title">
    <div className="comparison-card-heading">
      <div><span className="comparison-pipeline">PIPELINE 01</span><h3 id="comparison-rag-title">RAG</h3></div>
      <span className="comparison-status">{label(result.status)}</span>
    </div>
    {result.status === 'answered' && <div className="comparison-answer">{answerWithLinks(result.answer || '', result.citations, 'rag')}</div>}
    {result.status === 'insufficient_evidence' && <div className="comparison-insufficient"><h4>Insufficient evidence</h4><p>{result.message}</p></div>}
    <ErrorOutcome result={result} />
    <CitationList citations={result.citations} pipeline="rag" />
    <dl className="comparison-metrics">
      <div><dt>Elapsed</dt><dd>{formatDuration(result.elapsed_ms)}</dd></div>
      <div><dt>Grounding</dt><dd>{label(result.grounding_mode)}</dd></div>
      <div><dt>Strategy</dt><dd>{label(result.retrieval_strategy)}</dd></div>
      <div><dt>Retrieval</dt><dd>{formatDuration(result.retrieval_ms)}</dd></div>
      <div><dt>Generation</dt><dd>{formatDuration(result.generation_ms)}</dd></div>
      <div><dt>Chunks retrieved</dt><dd>{result.retrieved_chunk_count ?? 'Unavailable'}</dd></div>
      <div><dt>Chunks in context</dt><dd>{result.context_chunk_count ?? 'Unavailable'}</dd></div>
      {result.usage && <UsageMetrics prompt={result.usage.prompt_tokens} completion={result.usage.completion_tokens} />}
    </dl>
    <p className="comparison-provider">{result.provider ?? 'Provider unavailable'} / {result.model ?? 'model unavailable'}</p>
  </article>;
}

function AgentResultCard({ result, requestedStrategy }: { result: AgentComparison; requestedStrategy: Strategy | null }) {
  return <article className={`comparison-result-card outcome-${result.status}`} aria-labelledby="comparison-agent-title">
    <div className="comparison-card-heading">
      <div><span className="comparison-pipeline">PIPELINE 02</span><h3 id="comparison-agent-title">Agentic</h3></div>
      <span className="comparison-status">{label(result.status)}</span>
    </div>
    {(result.status === 'direct' || result.status === 'grounded')
      && <div className="comparison-answer">{answerWithLinks(result.answer || '', result.citations, 'agentic')}</div>}
    {result.status === 'insufficient_evidence' && <div className="comparison-insufficient"><h4>Insufficient evidence</h4><p>{result.message}</p></div>}
    <ErrorOutcome result={result} />
    <CitationList citations={result.citations} pipeline="agentic" />
    <dl className="comparison-metrics">
      <div><dt>Elapsed</dt><dd>{formatDuration(result.elapsed_ms)}</dd></div>
      <div><dt>Grounding</dt><dd>{label(result.grounding_mode)}</dd></div>
      <div><dt>Strategy</dt><dd>{label(requestedStrategy)}</dd></div>
      <div><dt>Route</dt><dd>{label(result.route)}</dd></div>
      <div><dt>Graph steps</dt><dd>{result.graph_steps ?? 'Unavailable'}</dd></div>
      <div><dt>Tool calls</dt><dd>{result.tool_call_count ?? 'Unavailable'}</dd></div>
      <div><dt>Execution budget</dt><dd>{formatDuration(result.execution_budget_ms)}</dd></div>
      <div><dt>Budget remaining</dt><dd>{formatDuration(result.remaining_budget_ms)}</dd></div>
      {result.usage && <UsageMetrics prompt={result.usage.prompt_tokens} completion={result.usage.completion_tokens} />}
    </dl>
    <p className="comparison-provider">{result.provider ?? 'Provider unavailable'} / {result.model ?? 'model unavailable'}</p>
    <details className="comparison-details">
      <summary>Tool activity ({result.tool_activity.length})</summary>
      {result.tool_activity.length ? <ol>{result.tool_activity.map(item => <li key={item.call_index}>
        <strong>Search {item.call_index}</strong>
        <span>{item.result_count} results / {item.evidence_added_count ?? 0} new evidence / top {item.top_k} / {label(item.retrieval_strategy)} / {formatDuration(item.duration_ms)}</span>
        <small>Query fingerprint: <code>{item.query_fingerprint ?? 'Unavailable'}</code>{item.remaining_budget_ms !== null && <> / remaining {formatDuration(item.remaining_budget_ms)}</>}</small>
      </li>)}</ol> : <p>No document-search tool was called.</p>}
    </details>
    <details className="comparison-details">
      <summary>Safe execution trace ({result.node_trace.length} nodes)</summary>
      {result.node_trace.length ? <ol>{result.node_trace.map((item, index) => <li key={`${item.node}-${index}`}>
        <strong><code>{item.node}</code></strong>
        <span>{item.outcome}{item.decision_action && <> / {label(item.decision_action)}</>}{item.decision_reason && <> / {label(item.decision_reason)}</>}{item.evidence_count !== null && <> / evidence {item.evidence_count}</>} / {formatDuration(item.duration_ms)}</span>
        {(item.error_code || item.validation_failure_code) && <small>{item.error_code && <>Error: <code>{item.error_code}</code></>}{item.validation_failure_code && <> / validation: <code>{item.validation_failure_code}</code></>}</small>}
      </li>)}</ol> : <p>No trace metadata was returned.</p>}
    </details>
  </article>;
}

export function ComparisonProgress() {
  return <div className="comparison-progress" role="status" aria-live="polite">
    <strong>Running both pipelines sequentially...</strong>
    <p>RAG runs first, followed by Agentic. Local CPU inference can take several minutes; this page does not impose a short client timeout.</p>
  </div>;
}

export function ComparisonResults({ result }: { result: ComparisonResponse }) {
  return <section className="comparison-results" aria-labelledby="comparison-results-title" aria-live="polite">
    <div className="comparison-results-heading">
      <div><span className="comparison-pipeline">SEQUENTIAL RUN</span><h2 id="comparison-results-title">Pipeline results</h2></div>
      <p>{formatDuration(result.total_elapsed_ms)} total / {label(result.document_scope)} / {label(result.requested_strategy)}</p>
    </div>
    <div className="comparison-quality-note"><strong>No automatic winner.</strong> Latency, tokens, citations, retrieved chunks, tool calls and graph steps describe pipeline activity. They are not answer-quality scores.</div>
    <div className="comparison-grid">
      <RagResultCard result={result.rag} />
      <AgentResultCard result={result.agentic} requestedStrategy={result.requested_strategy} />
    </div>
  </section>;
}

export function CompareQuestion({ libraryVersion }: { libraryVersion: number }) {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [total, setTotal] = useState(0);
  const [nextOffset, setNextOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [libraryError, setLibraryError] = useState('');
  const [reload, setReload] = useState(0);
  const [scope, setScope] = useState<ScopeMode>('all');
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [strategy, setStrategy] = useState<Strategy>('semantic');
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<ComparisonResponse | null>(null);
  const mounted = useRef(true);
  const comparisonRequest = useRef<AbortController | null>(null);
  const moreRequest = useRef<AbortController | null>(null);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      comparisonRequest.current?.abort();
      moreRequest.current?.abort();
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setLibraryError(''); setSelectedIds([]); setResult(null); setError('');
    requestJson('/api/documents?limit=100&offset=0', { signal: controller.signal }).then(page => {
      if (!isDocumentPage(page)) throw new Error('The API returned an unexpected document list.');
      if (!controller.signal.aborted) {
        setDocuments(page.documents); setTotal(page.total);
        setNextOffset(page.offset + page.documents.length);
      }
    }).catch(reason => {
      if (!controller.signal.aborted) setLibraryError(reason instanceof Error ? reason.message : 'The document library could not be loaded.');
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [libraryVersion, reload]);

  function clearResult() { setResult(null); setError(''); }

  async function loadMore() {
    if (moreRequest.current || loadingMore || busy) return;
    const controller = new AbortController(); moreRequest.current = controller;
    setLoadingMore(true); setLibraryError('');
    try {
      const page = await requestJson(`/api/documents?limit=100&offset=${nextOffset}`, { signal: controller.signal });
      if (!isDocumentPage(page)) throw new Error('The API returned an unexpected document list.');
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

  async function compare(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (comparisonRequest.current) return;
    clearResult();
    const value = question.trim();
    if (!value) { setError('Enter a question first.'); return; }
    if (value.length > 2000 || new TextEncoder().encode(value).length > 8000) {
      setError('Questions must be at most 2,000 characters and 8,000 encoded bytes.'); return;
    }
    if (scope === 'selected' && !selectedIds.length) {
      setError('Select at least one document, all documents, or no documents.'); return;
    }
    const documentIds = scope === 'all' ? null : scope === 'none' ? [] : selectedIds;
    const controller = new AbortController(); comparisonRequest.current = controller; setBusy(true);
    try {
      const response = await submitComparison({ question: value, document_ids: documentIds, strategy }, controller.signal);
      if (!controller.signal.aborted && mounted.current) setResult(response);
    } catch (reason) {
      if (mounted.current) {
        if (controller.signal.aborted) setError('Comparison cancelled in this browser. No result was applied; server-side local inference may continue until its existing execution limits stop it.');
        else setError(reason instanceof Error ? reason.message : 'The comparison request failed.');
      }
    } finally {
      if (comparisonRequest.current === controller) {
        comparisonRequest.current = null;
        if (mounted.current) setBusy(false);
      }
    }
  }

  function cancelComparison() { comparisonRequest.current?.abort(); }

  return <section className="comparison-panel" aria-labelledby="comparison-question-title" aria-busy={busy}>
    <div className="comparison-panel-heading">
      <div><h2 id="comparison-question-title">Compare RAG and Agentic</h2><p>Send one shared question and scope to the backend comparison engine. RAG runs first; Agentic runs second with independent results.</p></div>
      <button className="secondary-button" type="button" disabled={busy || loading} onClick={() => setReload(value => value + 1)}>Refresh documents</button>
    </div>
    {loading && <p role="status">Loading the shared document scope...</p>}
    {libraryError && <p className="comparison-warning" role="status">Document list unavailable: {libraryError} All-document and no-document requests remain available.</p>}
    <form onSubmit={compare}>
      <fieldset disabled={busy || loading}>
        <legend>Shared comparison inputs</legend>
        <div className="comparison-controls">
          <div>
            <span className="control-label">Document scope</span>
            <div className="comparison-scope-options">
              <label><input type="radio" name="comparison-scope" value="all" checked={scope === 'all'} onChange={() => { setScope('all'); clearResult(); }} /> All {total || ''} documents</label>
              <label><input type="radio" name="comparison-scope" value="selected" checked={scope === 'selected'} onChange={() => { setScope('selected'); clearResult(); }} /> Selected documents</label>
              <label><input type="radio" name="comparison-scope" value="none" checked={scope === 'none'} onChange={() => { setScope('none'); clearResult(); }} /> No documents</label>
            </div>
          </div>
          <label className="comparison-strategy" htmlFor="comparison-strategy">Retrieval strategy<select id="comparison-strategy" value={strategy} onChange={event => { setStrategy(event.target.value as Strategy); clearResult(); }}>{strategies.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
        </div>
        {scope === 'selected' && <div className="comparison-document-picker">
          <div className="document-options">{documents.map(document => <label key={document.id} className="document-option"><input type="checkbox" checked={selectedIds.includes(document.id)} onChange={event => { setSelectedIds(previous => event.target.checked ? [...previous, document.id] : previous.filter(id => id !== document.id)); clearResult(); }} /><span><strong>{document.filename}</strong><small>{document.page_count} pages / {document.chunk_count} chunks</small></span></label>)}</div>
          {!documents.length && <p>No documents are currently listed.</p>}
          <p>{selectedIds.length} selected / showing {documents.length} of {total}</p>
          {nextOffset < total && <button className="secondary-button" type="button" disabled={loadingMore} onClick={loadMore}>{loadingMore ? 'Loading more...' : 'Load more documents'}</button>}
        </div>}
        <label className="comparison-question-label" htmlFor="comparison-question">Shared question</label>
        <textarea id="comparison-question" rows={5} maxLength={2000} value={question} placeholder="Ask one question to run through both pipelines." onChange={event => { setQuestion(event.target.value); clearResult(); }} />
        <div className="comparison-actions"><span>{question.length} / 2000</span><button type="submit" disabled={busy}>{busy ? 'Comparison running...' : 'Run comparison'}</button></div>
      </fieldset>
      {busy && <button className="cancel-button" type="button" onClick={cancelComparison}>Cancel this request</button>}
    </form>
    {busy && <ComparisonProgress />}
    {error && <p className="comparison-error" role="alert">{error}</p>}
    {result && <ComparisonResults result={result} />}
  </section>;
}
