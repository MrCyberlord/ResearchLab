import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';

type AgentConfig = { provider: string; model: string; max_question_chars: number; graph_step_limit: number; execution_timeout_seconds: number; document_tools_available: false };
type NodeTrace = { node: string; duration_ms: number; outcome: string };
type Usage = { prompt_tokens: number | null; completion_tokens: number | null };
type Metadata = { route: 'direct' | 'document_tools_unavailable'; duration_ms: number; steps: number; node_trace: NodeTrace[]; usage: Usage | null };
type AgentResult = { answer: string; provider: string; model: string; document_tools_available: false; disclaimer: string; metadata: Metadata | null };

function record(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null; }
function count(value: unknown, minimum = 0): value is number { return typeof value === 'number' && Number.isInteger(value) && value >= minimum; }
function duration(value: unknown): value is number { return typeof value === 'number' && Number.isFinite(value) && value >= 0; }
function isConfig(value: unknown): value is AgentConfig {
  return record(value) && typeof value.provider === 'string' && typeof value.model === 'string'
    && count(value.max_question_chars, 1) && count(value.graph_step_limit, 1)
    && duration(value.execution_timeout_seconds) && value.document_tools_available === false;
}
function isUsage(value: unknown): value is Usage {
  return record(value) && (value.prompt_tokens === null || count(value.prompt_tokens))
    && (value.completion_tokens === null || count(value.completion_tokens));
}
function isTrace(value: unknown): value is NodeTrace {
  return record(value) && typeof value.node === 'string' && duration(value.duration_ms) && typeof value.outcome === 'string';
}
function isMetadata(value: unknown): value is Metadata {
  return record(value) && (value.route === 'direct' || value.route === 'document_tools_unavailable')
    && duration(value.duration_ms) && count(value.steps, 1) && Array.isArray(value.node_trace)
    && value.node_trace.length === value.steps && value.node_trace.every(isTrace)
    && (value.usage === null || isUsage(value.usage));
}
function isResult(value: unknown): value is AgentResult {
  return record(value) && typeof value.answer === 'string' && value.answer.trim().length > 0
    && typeof value.provider === 'string' && typeof value.model === 'string'
    && value.document_tools_available === false && typeof value.disclaimer === 'string'
    && (value.metadata === null || isMetadata(value.metadata));
}
async function requestJson(url: string, options: RequestInit = {}): Promise<unknown> {
  const response = await fetch(url, options);
  let body: unknown;
  try { body = await response.json(); } catch { throw new Error(`The API returned an unreadable response (HTTP ${response.status}).`); }
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

export function AgentQuestion() {
  const [config, setConfig] = useState<AgentConfig | null>(null);
  const [configError, setConfigError] = useState('');
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<AgentResult | null>(null);
  const [reload, setReload] = useState(0);
  const request = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setConfig(null); setConfigError('');
    requestJson('/api/agent/config', { signal: controller.signal }).then(value => {
      if (!isConfig(value)) throw new Error('The API returned an unexpected agent configuration.');
      if (!controller.signal.aborted) setConfig(value);
    }).catch(reason => {
      if (!controller.signal.aborted) setConfigError(reason instanceof Error ? reason.message : 'Agent configuration could not be loaded.');
    });
    return () => { controller.abort(); request.current?.abort(); };
  }, [reload]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (request.current) return;
    setError(''); setResult(null);
    const value = question.trim();
    if (!config) { setError('Load the agent configuration before submitting a question.'); return; }
    if (!value) { setError('Enter a question first.'); return; }
    if (value.length > config.max_question_chars) { setError(`Questions must be at most ${config.max_question_chars} characters.`); return; }
    const controller = new AbortController(); request.current = controller; setBusy(true);
    try {
      const data = await requestJson('/api/agent/questions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
        body: JSON.stringify({ question: value, include_metadata: true }),
      });
      if (!isResult(data)) throw new Error('The API returned an unexpected agent response.');
      if (!controller.signal.aborted) setResult(data);
    } catch (reason) {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'The agent could not respond.');
    } finally {
      if (request.current === controller) { request.current = null; setBusy(false); }
    }
  }

  return <section className="agent-panel" aria-labelledby="agent-question-title" aria-busy={busy}>
    <div className="agent-panel-heading"><div><h2 id="agent-question-title">Run the bounded agent</h2><p>Ask a general question. A real LangGraph route invokes the configured local model once and returns an optional node trace.</p></div></div>
    <div className="agent-limitation"><strong>Foundation scope:</strong> this agent cannot search uploaded PDFs, call tools, browse, remember prior turns or provide document citations yet.</div>
    {config && <p className="provider-label">Configured provider: {config.provider} / {config.model}. Maximum {config.graph_step_limit} graph steps; {config.execution_timeout_seconds}s provider timeout.</p>}
    {!config && !configError && <p role="status">Loading agent configuration...</p>}
    {configError && <p className="agent-error" role="alert">{configError} <button className="text-button" type="button" onClick={() => setReload(value => value + 1)}>Retry</button></p>}
    <form onSubmit={submit}><label htmlFor="agent-question">Your question</label><textarea id="agent-question" value={question} rows={4} maxLength={config?.max_question_chars} placeholder="Ask a general question for the local model." onChange={event => { setQuestion(event.target.value); setResult(null); setError(''); }} disabled={busy || !config} />
      <div className="agent-actions"><span>{question.length} / {config?.max_question_chars ?? 2000}</span><button type="submit" disabled={busy || !config}>{busy ? 'Routing and generating...' : 'Run agent'}</button></div></form>
    <div role="status" aria-live="polite">{busy && <p>Executing the bounded graph and waiting for the configured local model...</p>}</div>
    {error && <p className="agent-error" role="alert">{error}</p>}
    {result && <div className="agent-result" aria-live="polite"><h3>Agent response</h3><div className="agent-answer">{result.answer}</div><p className="agent-disclaimer">{result.disclaimer}</p>
      {result.metadata && <details className="agent-trace"><summary>Execution trace: {result.metadata.steps} nodes / {result.metadata.duration_ms.toFixed(0)} ms</summary><ol>{result.metadata.node_trace.map((item, index) => <li key={`${item.node}-${index}`}><code>{item.node}</code><span>{item.outcome} / {item.duration_ms.toFixed(1)} ms</span></li>)}</ol><p>Route: <code>{result.metadata.route}</code>{result.metadata.usage && <> / Prompt tokens: {result.metadata.usage.prompt_tokens ?? 'unavailable'} / Output tokens: {result.metadata.usage.completion_tokens ?? 'unavailable'}</>}</p></details>}
      <p className="agent-provider">{result.provider} / {result.model}</p></div>}
  </section>;
}
