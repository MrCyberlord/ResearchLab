import { useEffect, useRef, useState } from 'react';
import type { FormEvent, ReactNode } from 'react';

type Strategy = 'semantic' | 'hybrid' | 'hybrid_rerank';
type AgentConfig = {
  provider: string; model: string; max_question_chars: number; graph_step_limit: number;
  execution_timeout_seconds: number; max_tool_calls: number; tool_top_k: number;
  max_evidence_chunks: number; default_strategy: Strategy;
  available_strategies: Strategy[]; document_tools_available: true;
};
type DocumentItem = { id: string; filename: string; page_count: number; chunk_count: number };
type DocumentPage = { documents: DocumentItem[]; total: number; limit: number; offset: number };
type NodeTrace = { node: string; duration_ms: number; outcome: string };
type ToolActivity = { call_index: number; query: string; top_k: number; result_count: number; duration_ms: number; outcome: 'results' | 'empty' };
type Usage = { prompt_tokens: number | null; completion_tokens: number | null };
type Citation = { source_id: string; chunk_id: number; document_id: string; filename: string; page_number: number; chunk_order: number; quote: string; text: string };
type Metadata = { route: 'direct' | 'document_research'; duration_ms: number; steps: number; tool_call_count: number; node_trace: NodeTrace[]; tool_activity: ToolActivity[]; usage: Usage | null };
type AgentResult = { status: 'direct' | 'grounded' | 'insufficient_evidence'; answer: string | null; message: string | null; citations: Citation[]; provider: string; model: string; document_tools_available: true; disclaimer: string; metadata: Metadata | null };

function record(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null; }
function count(value: unknown, minimum = 0): value is number { return typeof value === 'number' && Number.isInteger(value) && value >= minimum; }
function duration(value: unknown): value is number { return typeof value === 'number' && Number.isFinite(value) && value >= 0; }
function isStrategy(value: unknown): value is Strategy { return value === 'semantic' || value === 'hybrid' || value === 'hybrid_rerank'; }
function isDocument(value: unknown): value is DocumentItem { return record(value) && typeof value.id === 'string' && typeof value.filename === 'string' && count(value.page_count, 1) && count(value.chunk_count, 1); }
function isPage(value: unknown): value is DocumentPage { return record(value) && Array.isArray(value.documents) && value.documents.every(isDocument) && count(value.total) && count(value.limit, 1) && count(value.offset); }
function isConfig(value: unknown): value is AgentConfig {
  return record(value) && typeof value.provider === 'string' && typeof value.model === 'string'
    && count(value.max_question_chars, 1) && count(value.graph_step_limit, 1)
    && duration(value.execution_timeout_seconds) && count(value.max_tool_calls, 1)
    && count(value.tool_top_k, 1) && count(value.max_evidence_chunks, 1)
    && isStrategy(value.default_strategy) && Array.isArray(value.available_strategies)
    && value.available_strategies.every(isStrategy) && value.document_tools_available === true;
}
function isUsage(value: unknown): value is Usage { return record(value) && (value.prompt_tokens === null || count(value.prompt_tokens)) && (value.completion_tokens === null || count(value.completion_tokens)); }
function isTrace(value: unknown): value is NodeTrace { return record(value) && typeof value.node === 'string' && duration(value.duration_ms) && typeof value.outcome === 'string'; }
function isTool(value: unknown): value is ToolActivity { return record(value) && count(value.call_index, 1) && typeof value.query === 'string' && count(value.top_k, 1) && count(value.result_count) && duration(value.duration_ms) && (value.outcome === 'results' || value.outcome === 'empty'); }
function isCitation(value: unknown): value is Citation { return record(value) && typeof value.source_id === 'string' && /^S[1-9]\d*$/.test(value.source_id) && count(value.chunk_id, 1) && typeof value.document_id === 'string' && typeof value.filename === 'string' && count(value.page_number, 1) && count(value.chunk_order) && typeof value.quote === 'string' && typeof value.text === 'string'; }
function isMetadata(value: unknown): value is Metadata {
  return record(value) && (value.route === 'direct' || value.route === 'document_research')
    && duration(value.duration_ms) && count(value.steps, 1) && count(value.tool_call_count)
    && Array.isArray(value.node_trace) && value.node_trace.length === value.steps && value.node_trace.every(isTrace)
    && Array.isArray(value.tool_activity) && value.tool_activity.length === value.tool_call_count && value.tool_activity.every(isTool)
    && (value.usage === null || isUsage(value.usage));
}
function isResult(value: unknown): value is AgentResult {
  if (!record(value) || !Array.isArray(value.citations) || !value.citations.every(isCitation)
      || typeof value.provider !== 'string' || typeof value.model !== 'string'
      || value.document_tools_available !== true || typeof value.disclaimer !== 'string'
      || !(value.metadata === null || isMetadata(value.metadata))) return false;
  if (value.status === 'direct') return typeof value.answer === 'string' && !!value.answer.trim() && value.citations.length === 0 && (value.message === null || typeof value.message === 'string');
  if (value.status === 'grounded') return typeof value.answer === 'string' && !!value.answer.trim() && value.citations.length > 0 && (value.message === null || typeof value.message === 'string');
  return value.status === 'insufficient_evidence' && value.answer === null && typeof value.message === 'string' && value.citations.length === 0;
}
async function requestJson(url: string, options: RequestInit = {}): Promise<unknown> {
  const response = await fetch(url, options); let body: unknown;
  try { body = await response.json(); } catch { throw new Error(`The API returned an unreadable response (HTTP ${response.status}).`); }
  if (!response.ok) {
    let message = `Request failed (HTTP ${response.status}).`;
    if (record(body)) { const detail = body.detail; if (typeof detail === 'string') message = detail; else if (record(detail) && typeof detail.message === 'string') message = detail.message; else if (Array.isArray(detail)) { const values = detail.filter(record).map(item => item.msg).filter((item): item is string => typeof item === 'string'); if (values.length) message = values.join('; '); } }
    throw new Error(message);
  }
  return body;
}
function answerWithLinks(answer: string, citations: Citation[]): ReactNode[] {
  const sources = new Set(citations.map(item => item.source_id));
  return answer.split(/(\[S[1-9]\d*\])/g).map((part, index) => {
    const source = part.slice(1, -1);
    return /^\[S[1-9]\d*\]$/.test(part) && sources.has(source)
      ? <a key={index} href={`#agent-source-${source}`}>{part}</a> : <span key={index}>{part}</span>;
  });
}

export function AgentQuestion() {
  const [config, setConfig] = useState<AgentConfig | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [total, setTotal] = useState(0);
  const [configError, setConfigError] = useState('');
  const [documentError, setDocumentError] = useState('');
  const [reload, setReload] = useState(0);
  const [allDocuments, setAllDocuments] = useState(true);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [strategy, setStrategy] = useState<Strategy>('semantic');
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<AgentResult | null>(null);
  const request = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController(); setConfig(null); setConfigError(''); setDocumentError(''); setSelectedIds([]);
    requestJson('/api/agent/config', { signal: controller.signal }).then(settings => {
      if (!isConfig(settings)) throw new Error('The API returned an unexpected Agentic configuration.');
      if (!controller.signal.aborted) { setConfig(settings); setStrategy(settings.default_strategy); }
    }).catch(reason => { if (!controller.signal.aborted) setConfigError(reason instanceof Error ? reason.message : 'Agentic configuration could not be loaded.'); });
    requestJson('/api/documents?limit=100&offset=0', { signal: controller.signal }).then(page => {
      if (!isPage(page)) throw new Error('The API returned an unexpected document list.');
      if (!controller.signal.aborted) { setDocuments(page.documents); setTotal(page.total); }
    }).catch(reason => { if (!controller.signal.aborted) setDocumentError(reason instanceof Error ? reason.message : 'The document list could not be loaded.'); });
    return () => { controller.abort(); request.current?.abort(); };
  }, [reload]);

  function clearResult() { setResult(null); setError(''); }
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); if (request.current) return; clearResult(); const value = question.trim();
    if (!config) { setError('Load Agentic configuration before submitting.'); return; }
    if (!value) { setError('Enter a question first.'); return; }
    if (value.length > config.max_question_chars) { setError(`Questions must be at most ${config.max_question_chars} characters.`); return; }
    if (!allDocuments && !selectedIds.length) { setError('Select at least one document or choose all documents.'); return; }
    const controller = new AbortController(); request.current = controller; setBusy(true);
    try {
      const data = await requestJson('/api/agent/questions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal, body: JSON.stringify({ question: value, document_ids: allDocuments ? null : selectedIds, strategy, include_metadata: true }) });
      if (!isResult(data)) throw new Error('The API returned an unexpected Agentic response.');
      if (!controller.signal.aborted) setResult(data);
    } catch (reason) { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'The agent could not respond.'); }
    finally { if (request.current === controller) { request.current = null; setBusy(false); } }
  }

  return <section className="agent-panel" aria-labelledby="agent-question-title" aria-busy={busy}>
    <div className="agent-panel-heading"><div><h2 id="agent-question-title">Run the document research agent</h2><p>The model decides whether to answer generally or use bounded document search. Grounded answers include validated source quotes.</p></div><button className="secondary-button" type="button" disabled={busy} onClick={() => setReload(value => value + 1)}>Refresh documents</button></div>
    <div className="agent-limitation"><strong>Tool boundary:</strong> document search is available. Web browsing, external tools, memory and autonomous open-ended loops are not.</div>
    {config && <p className="provider-label">{config.provider} / {config.model} · up to {config.max_tool_calls} searches · {config.graph_step_limit} graph steps · {config.execution_timeout_seconds}s total deadline</p>}
    {!config && !configError && <p role="status">Loading Agentic configuration...</p>}
    {configError && <p className="agent-error" role="alert">{configError} <button className="text-button" type="button" onClick={() => setReload(value => value + 1)}>Retry</button></p>}
    {documentError && <p className="agent-warning" role="status">Document list unavailable: {documentError} General questions can still use the direct path.</p>}
    <form onSubmit={submit}>
      <fieldset disabled={busy || !config}><legend>Document tool scope</legend><label className="scope-option"><input type="checkbox" checked={allDocuments} onChange={event => { setAllDocuments(event.target.checked); clearResult(); }} /> Search all {total || ''} documents when the model calls the tool</label>
        {!allDocuments && <div className="agent-document-options">{documents.map(document => <label key={document.id}><input type="checkbox" checked={selectedIds.includes(document.id)} onChange={event => { setSelectedIds(previous => event.target.checked ? [...previous, document.id] : previous.filter(id => id !== document.id)); clearResult(); }} /><span><strong>{document.filename}</strong><small>{document.page_count} pages / {document.chunk_count} chunks</small></span></label>)}{!documents.length && <p>No documents are available.</p>}</div>}
        <label className="agent-strategy" htmlFor="agent-strategy">Document retrieval strategy <select id="agent-strategy" value={strategy} onChange={event => { setStrategy(event.target.value as Strategy); clearResult(); }}>{config?.available_strategies.map(item => <option key={item} value={item}>{item}</option>)}</select></label>
      </fieldset>
      <label htmlFor="agent-question">Your question</label><textarea id="agent-question" value={question} rows={4} maxLength={config?.max_question_chars} placeholder="Ask generally, or ask the agent to research your uploaded documents." onChange={event => { setQuestion(event.target.value); clearResult(); }} disabled={busy || !config} />
      <div className="agent-actions"><span>{question.length} / {config?.max_question_chars ?? 2000}</span><button type="submit" disabled={busy || !config}>{busy ? 'Deciding, searching and generating...' : 'Run agent'}</button></div>
    </form>
    <div role="status" aria-live="polite">{busy && <p>The bounded graph is deciding whether to call document search. Local generation can take a moment.</p>}</div>
    {error && <p className="agent-error" role="alert">{error}</p>}
    {result && <div className="agent-result" aria-live="polite"><h3>{result.status === 'grounded' ? 'Document-grounded answer' : result.status === 'direct' ? 'General answer' : 'Insufficient evidence'}</h3>{result.answer ? <div className="agent-answer">{answerWithLinks(result.answer, result.citations)}</div> : <p>{result.message}</p>}<p className="agent-disclaimer">{result.disclaimer}</p>
      {!!result.citations.length && <><h3>Supporting sources</h3><div className="agent-citations">{result.citations.map(item => <article key={item.source_id} id={`agent-source-${item.source_id}`}><h4>[{item.source_id}] {item.filename} / page {item.page_number}</h4><blockquote>{item.quote}</blockquote><details><summary>Supporting chunk {item.chunk_id}</summary><p>Document <code>{item.document_id}</code> / chunk order {item.chunk_order}</p><pre>{item.text}</pre></details></article>)}</div></>}
      {result.metadata && <><div className="agent-tool-summary"><h3>Tool activity</h3>{result.metadata.tool_activity.length ? <ol>{result.metadata.tool_activity.map(item => <li key={item.call_index}><strong>Search {item.call_index}:</strong> {item.query}<span>{item.result_count} results / top {item.top_k} / {item.duration_ms.toFixed(0)} ms</span></li>)}</ol> : <p>No document tool was called.</p>}</div><details className="agent-trace"><summary>Execution trace: {result.metadata.steps} nodes / {result.metadata.duration_ms.toFixed(0)} ms</summary><ol>{result.metadata.node_trace.map((item, index) => <li key={`${item.node}-${index}`}><code>{item.node}</code><span>{item.outcome} / {item.duration_ms.toFixed(1)} ms</span></li>)}</ol><p>Route: <code>{result.metadata.route}</code> / Tool calls: {result.metadata.tool_call_count}{result.metadata.usage && <> / Prompt tokens: {result.metadata.usage.prompt_tokens ?? 'unavailable'} / Output tokens: {result.metadata.usage.completion_tokens ?? 'unavailable'}</>}</p></details></>}
      <p className="agent-provider">{result.provider} / {result.model}</p></div>}
  </section>;
}
