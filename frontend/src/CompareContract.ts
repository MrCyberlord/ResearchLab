export type Strategy = 'semantic' | 'hybrid' | 'hybrid_rerank';
export type DocumentScope = 'all_documents' | 'selected_documents' | 'no_documents';
export type ScopeMode = 'all' | 'selected' | 'none';

export type DocumentItem = {
  id: string;
  filename: string;
  page_count: number;
  chunk_count: number;
};

export type DocumentPage = {
  documents: DocumentItem[];
  total: number;
  limit: number;
  offset: number;
};

export type ComparisonError = {
  code: string;
  category: 'validation' | 'configuration' | 'embedding' | 'retrieval' | 'provider' | 'timeout' | 'execution' | 'internal';
  message: string;
  retryable: boolean;
};

export type Citation = {
  source_id: string;
  chunk_id: number;
  document_id: string;
  filename: string;
  page_number: number;
  chunk_order: number;
  quote: string;
};

export type Usage = { prompt_tokens: number | null; completion_tokens: number | null };

export type NodeTrace = {
  node: string;
  duration_ms: number;
  outcome: string;
  decision_action: 'answer_direct' | 'search_documents' | 'answer_grounded' | 'insufficient_evidence' | null;
  decision_scope: 'general_knowledge' | 'document_research' | null;
  decision_reason: 'general_question' | 'evidence_needed' | 'refine_search' | 'evidence_supports_answer' | 'evidence_does_not_support_answer' | 'search_budget_exhausted' | null;
  evidence_count: number | null;
  remaining_budget_ms: number | null;
  error_type: string | null;
  error_code: string | null;
  validation_retry_count: number | null;
  validation_failure_code: string | null;
};

export type ToolActivity = {
  call_index: number;
  top_k: number;
  result_count: number;
  duration_ms: number;
  outcome: 'results' | 'empty';
  query_fingerprint: string | null;
  retrieval_strategy: Strategy | null;
  document_filter_count: number | null;
  evidence_added_count: number | null;
  remaining_budget_ms: number | null;
};

export type RagComparison = {
  pipeline: 'rag';
  status: 'answered' | 'insufficient_evidence' | 'error';
  grounding_mode: 'document_grounded';
  document_scope: DocumentScope;
  answer: string | null;
  message: string | null;
  citations: Citation[];
  elapsed_ms: number;
  error: ComparisonError | null;
  provider: string | null;
  model: string | null;
  retrieval_strategy: Strategy | null;
  retrieval_ms: number | null;
  generation_ms: number | null;
  retrieved_chunk_count: number | null;
  context_chunk_count: number | null;
  usage: Usage | null;
};

export type AgentComparison = {
  pipeline: 'agentic';
  status: 'direct' | 'grounded' | 'insufficient_evidence' | 'error';
  grounding_mode: 'general_knowledge' | 'document_grounded' | 'insufficient_document_evidence' | 'unknown';
  document_scope: DocumentScope;
  route: 'direct' | 'document_research' | null;
  answer: string | null;
  message: string | null;
  citations: Citation[];
  elapsed_ms: number;
  error: ComparisonError | null;
  provider: string | null;
  model: string | null;
  graph_steps: number | null;
  tool_call_count: number | null;
  node_trace: NodeTrace[];
  tool_activity: ToolActivity[];
  usage: Usage | null;
  execution_budget_ms: number | null;
  remaining_budget_ms: number | null;
};

export type ComparisonResponse = {
  execution_mode: 'sequential';
  total_elapsed_ms: number;
  document_scope: DocumentScope;
  requested_strategy: Strategy | null;
  activity_metrics_are_quality_measures: false;
  rag: RagComparison;
  agentic: AgentComparison;
};

export type ComparisonInput = {
  question: string;
  document_ids: string[] | null;
  strategy: Strategy;
};

type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function count(value: unknown, minimum = 0): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= minimum;
}

function duration(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0;
}

function nullableString(value: unknown): value is string | null {
  return value === null || typeof value === 'string';
}

export function isStrategy(value: unknown): value is Strategy {
  return value === 'semantic' || value === 'hybrid' || value === 'hybrid_rerank';
}

function isScope(value: unknown): value is DocumentScope {
  return value === 'all_documents' || value === 'selected_documents' || value === 'no_documents';
}

function isDocument(value: unknown): value is DocumentItem {
  return record(value) && typeof value.id === 'string' && typeof value.filename === 'string'
    && count(value.page_count, 1) && count(value.chunk_count, 1);
}

export function isDocumentPage(value: unknown): value is DocumentPage {
  return record(value) && Array.isArray(value.documents) && value.documents.every(isDocument)
    && count(value.total) && count(value.limit, 1) && count(value.offset);
}

function isError(value: unknown): value is ComparisonError {
  return record(value) && typeof value.code === 'string' && value.code.length > 0
    && (value.category === 'validation' || value.category === 'configuration'
      || value.category === 'embedding' || value.category === 'retrieval'
      || value.category === 'provider' || value.category === 'timeout'
      || value.category === 'execution' || value.category === 'internal')
    && typeof value.message === 'string' && value.message.length > 0
    && typeof value.retryable === 'boolean';
}

function isCitation(value: unknown): value is Citation {
  return record(value) && typeof value.source_id === 'string' && /^S[1-9]\d*$/.test(value.source_id)
    && count(value.chunk_id, 1) && typeof value.document_id === 'string'
    && typeof value.filename === 'string' && value.filename.length > 0
    && count(value.page_number, 1) && count(value.chunk_order)
    && typeof value.quote === 'string' && value.quote.length > 0;
}

function isUsage(value: unknown): value is Usage {
  return record(value) && (value.prompt_tokens === null || count(value.prompt_tokens))
    && (value.completion_tokens === null || count(value.completion_tokens));
}

function isNullableDuration(value: unknown): value is number | null {
  return value === null || duration(value);
}

function isNullableCount(value: unknown): value is number | null {
  return value === null || count(value);
}

function isTrace(value: unknown): value is NodeTrace {
  const actions = ['answer_direct', 'search_documents', 'answer_grounded', 'insufficient_evidence'];
  const scopes = ['general_knowledge', 'document_research'];
  const reasons = ['general_question', 'evidence_needed', 'refine_search', 'evidence_supports_answer', 'evidence_does_not_support_answer', 'search_budget_exhausted'];
  return record(value) && typeof value.node === 'string' && duration(value.duration_ms)
    && typeof value.outcome === 'string'
    && (value.decision_action === null || actions.includes(String(value.decision_action)))
    && (value.decision_scope === null || scopes.includes(String(value.decision_scope)))
    && (value.decision_reason === null || reasons.includes(String(value.decision_reason)))
    && isNullableCount(value.evidence_count) && isNullableDuration(value.remaining_budget_ms)
    && nullableString(value.error_type) && nullableString(value.error_code)
    && isNullableCount(value.validation_retry_count)
    && nullableString(value.validation_failure_code);
}

function isToolActivity(value: unknown): value is ToolActivity {
  return record(value) && count(value.call_index, 1) && count(value.top_k, 1)
    && count(value.result_count) && duration(value.duration_ms)
    && (value.outcome === 'results' || value.outcome === 'empty')
    && (value.query_fingerprint === null
      || (typeof value.query_fingerprint === 'string' && /^[0-9a-f]{16}$/.test(value.query_fingerprint)))
    && (value.retrieval_strategy === null || isStrategy(value.retrieval_strategy))
    && isNullableCount(value.document_filter_count)
    && isNullableCount(value.evidence_added_count)
    && isNullableDuration(value.remaining_budget_ms);
}

function hasCitations(value: Record<string, unknown>): value is Record<string, unknown> & { citations: Citation[] } {
  return Array.isArray(value.citations) && value.citations.every(isCitation)
    && new Set(value.citations.map(item => item.source_id)).size === value.citations.length;
}

function isRag(value: unknown): value is RagComparison {
  if (!record(value) || !hasCitations(value) || value.pipeline !== 'rag'
      || value.grounding_mode !== 'document_grounded' || !isScope(value.document_scope)
      || !duration(value.elapsed_ms) || !nullableString(value.answer) || !nullableString(value.message)
      || !(value.error === null || isError(value.error)) || !nullableString(value.provider)
      || !nullableString(value.model) || !(value.retrieval_strategy === null || isStrategy(value.retrieval_strategy))
      || !isNullableDuration(value.retrieval_ms) || !isNullableDuration(value.generation_ms)
      || !isNullableCount(value.retrieved_chunk_count) || !isNullableCount(value.context_chunk_count)
      || !(value.usage === null || isUsage(value.usage))) return false;
  if (value.status === 'answered') return typeof value.answer === 'string' && !!value.answer.trim()
    && value.citations.length > 0 && value.error === null;
  if (value.status === 'insufficient_evidence') return value.answer === null
    && typeof value.message === 'string' && !!value.message.trim() && value.citations.length === 0
    && value.error === null;
  return value.status === 'error' && value.answer === null && value.citations.length === 0
    && value.error !== null;
}

function isAgent(value: unknown): value is AgentComparison {
  if (!record(value) || !hasCitations(value) || value.pipeline !== 'agentic'
      || !isScope(value.document_scope) || !duration(value.elapsed_ms)
      || !nullableString(value.answer) || !nullableString(value.message)
      || !(value.error === null || isError(value.error)) || !nullableString(value.provider)
      || !nullableString(value.model) || !(value.route === null || value.route === 'direct' || value.route === 'document_research')
      || !isNullableCount(value.graph_steps) || !isNullableCount(value.tool_call_count)
      || !Array.isArray(value.node_trace) || !value.node_trace.every(isTrace)
      || !Array.isArray(value.tool_activity) || !value.tool_activity.every(isToolActivity)
      || !(value.usage === null || isUsage(value.usage))
      || !isNullableDuration(value.execution_budget_ms) || !isNullableDuration(value.remaining_budget_ms)) return false;
  if (value.graph_steps !== null && value.graph_steps !== value.node_trace.length) return false;
  if (value.tool_call_count !== null && value.tool_call_count !== value.tool_activity.length) return false;
  if (value.status === 'direct') return value.grounding_mode === 'general_knowledge'
    && typeof value.answer === 'string' && !!value.answer.trim() && value.citations.length === 0
    && value.error === null;
  if (value.status === 'grounded') return value.grounding_mode === 'document_grounded'
    && typeof value.answer === 'string' && !!value.answer.trim() && value.citations.length > 0
    && value.error === null;
  if (value.status === 'insufficient_evidence') return value.grounding_mode === 'insufficient_document_evidence'
    && value.answer === null && typeof value.message === 'string' && !!value.message.trim()
    && value.citations.length === 0 && value.error === null;
  return value.status === 'error' && value.grounding_mode === 'unknown'
    && value.answer === null && value.citations.length === 0 && value.error !== null;
}

export function isComparisonResponse(value: unknown): value is ComparisonResponse {
  return record(value) && value.execution_mode === 'sequential' && duration(value.total_elapsed_ms)
    && isScope(value.document_scope) && (value.requested_strategy === null || isStrategy(value.requested_strategy))
    && value.activity_metrics_are_quality_measures === false && isRag(value.rag)
    && isAgent(value.agentic);
}

export async function requestJson(url: string, options: RequestInit = {}, fetcher: FetchLike = fetch): Promise<unknown> {
  const response = await fetcher(url, options);
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new Error(`The API returned an unreadable response (HTTP ${response.status}).`);
  }
  if (!response.ok) {
    let message = `Request failed (HTTP ${response.status}).`;
    if (record(body)) {
      const detail = body.detail;
      if (typeof detail === 'string') message = detail;
      else if (record(detail) && typeof detail.message === 'string') message = detail.message;
      else if (Array.isArray(detail)) {
        const messages = detail.filter(record).map(item => item.msg)
          .filter((item): item is string => typeof item === 'string');
        if (messages.length) message = messages.join('; ');
      }
    }
    throw new Error(message);
  }
  return body;
}

export async function submitComparison(
  input: ComparisonInput,
  signal: AbortSignal,
  fetcher: FetchLike = fetch,
): Promise<ComparisonResponse> {
  const body = await requestJson('/api/compare/questions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
    signal,
  }, fetcher);
  if (!isComparisonResponse(body)) throw new Error('The API returned an unexpected comparison response.');
  return body;
}
