export type EvaluationRun = { run_id: string; created_at_utc: string };
export type EvaluationLatency = { sample_count: number; p50_ms: number; p95_ms: number };
export type RagMetric = {
  k: number; recall_at_k: number; precision_at_k: number; mrr_at_k: number;
  hit_rate_at_k: number; answerable_question_count: number;
};
export type RagStrategy = {
  strategy: 'semantic' | 'hybrid' | 'hybrid_rerank'; reference_only: boolean;
  metrics: RagMetric[]; latency: EvaluationLatency;
};
export type MetricDefinition = { key: string; label: string; definition: string; denominator: string };
export type RagEvaluationSummary = {
  dataset: {
    version: string; title: string; license: string; document_count: number;
    question_count: number; answerable_question_count: number;
    unanswerable_question_count: number; sha256: string;
  };
  baseline_run: EvaluationRun;
  strategy_reference_run: EvaluationRun | null;
  model: { provider: string; model: string; revision: string; dimension: number; normalization: string };
  settings: {
    default_strategy: 'semantic' | 'hybrid' | 'hybrid_rerank'; k_values: number[];
    min_similarity: number; distance: string; chunk_size: number; chunk_overlap: number;
    semantic_candidate_count: number | null; keyword_candidate_count: number | null;
    rrf_fusion_constant: number | null;
  };
  strategies: RagStrategy[];
  metric_definitions: MetricDefinition[];
  comparability_note: string;
};
export type AgentMetric = {
  key: string; label: string; value: number | null; denominator_name: string;
  denominator: number; definition: string;
};
export type Distribution = { sample_count: number; mean: number; minimum: number; maximum: number };
export type AgentEvaluationSummary = {
  dataset: {
    version: string; title: string; license: string; document_count: number;
    question_count: number; repeats: number; attempted_run_count: number; sha256: string;
  };
  run: EvaluationRun;
  model: { provider: string; model: string; temperature: number };
  settings: {
    scope_prompt_version: string; decision_prompt_version: string;
    decision_schema_version: string; generation_schema_version: string;
    graph_step_limit: number; max_tool_calls: number; tool_top_k: number;
    execution_timeout_seconds: number; retrieval_strategy: 'semantic' | 'hybrid' | 'hybrid_rerank';
    min_similarity: number; embedding_model: string; embedding_revision: string;
  };
  denominators: {
    attempted_runs: number; completed_runs: number; failed_runs: number;
    expected_grounded_runs: number; expected_abstention_runs: number;
    citation_count: number; expected_relevant_reference_count: number; questions: number;
  };
  metrics: AgentMetric[];
  latency: EvaluationLatency;
  activity: {
    tool_calls: Distribution; graph_steps: Distribution;
    stable_question_rate: number | null; questions_with_variation: number;
  };
  failures: { count: number; rate: number; configured_timeout_overrun_count: number };
  usage: { samples_with_usage: number; prompt_tokens: number; completion_tokens: number };
  limitations: { citation_check: string; factual_entailment_measured: false; sample_size: string };
  evidence_observation: null | {
    run: EvaluationRun; sample_size: number; expected_status: string; observed_status: string;
    retrieved_result_count: number; evidence_count: number; tool_call_count: number;
    finding: string; caution: string;
  };
  comparability_note: string;
};
export type EvaluationSummary = {
  schema_version: 1;
  status: 'complete' | 'partial' | 'unavailable';
  artifacts: { artifact_id: string; state: 'loaded' | 'missing' | 'invalid' | 'incomplete' | 'incompatible' }[];
  issues: { artifact_id: string; code: string; message: string }[];
  rag: RagEvaluationSummary | null;
  agentic: AgentEvaluationSummary | null;
  research_findings: { source: string; title: string; summary: string; caution: string }[];
  cross_dataset_comparison_allowed: false;
  generated_from_versioned_artifacts_only: true;
};

type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}
function integer(value: unknown, minimum = 0): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= minimum;
}
function number(value: unknown, minimum = 0): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= minimum;
}
function ratio(value: unknown): value is number {
  return number(value) && value <= 1;
}
function text(value: unknown): value is string { return typeof value === 'string' && value.length > 0; }
function strategy(value: unknown): value is 'semantic' | 'hybrid' | 'hybrid_rerank' {
  return value === 'semantic' || value === 'hybrid' || value === 'hybrid_rerank';
}
function nullableInteger(value: unknown, minimum = 0): value is number | null {
  return value === null || integer(value, minimum);
}
function sha256(value: unknown): value is string {
  return typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
}
function run(value: unknown): value is EvaluationRun {
  return record(value) && text(value.run_id) && text(value.created_at_utc);
}
function latency(value: unknown): value is EvaluationLatency {
  return record(value) && integer(value.sample_count, 1) && number(value.p50_ms) && number(value.p95_ms);
}
function distribution(value: unknown): value is Distribution {
  return record(value) && integer(value.sample_count) && number(value.mean)
    && integer(value.minimum) && integer(value.maximum);
}
function ragMetric(value: unknown): value is RagMetric {
  return record(value) && integer(value.k, 1) && ratio(value.recall_at_k)
    && ratio(value.precision_at_k) && ratio(value.mrr_at_k) && ratio(value.hit_rate_at_k)
    && integer(value.answerable_question_count, 1);
}
function ragStrategy(value: unknown): value is RagStrategy {
  return record(value) && (value.strategy === 'semantic' || value.strategy === 'hybrid' || value.strategy === 'hybrid_rerank')
    && typeof value.reference_only === 'boolean' && Array.isArray(value.metrics)
    && value.metrics.every(ragMetric) && latency(value.latency);
}
function ragSummary(value: unknown): value is RagEvaluationSummary {
  if (!record(value)) return false;
  const dataset = value.dataset; const model = value.model; const settings = value.settings;
  return record(dataset) && text(dataset.version) && text(dataset.title) && text(dataset.license)
    && integer(dataset.document_count, 1) && integer(dataset.question_count, 1)
    && integer(dataset.answerable_question_count, 1) && integer(dataset.unanswerable_question_count)
    && sha256(dataset.sha256) && run(value.baseline_run)
    && (value.strategy_reference_run === null || run(value.strategy_reference_run))
    && record(model) && text(model.provider) && text(model.model) && text(model.revision)
    && integer(model.dimension, 1) && text(model.normalization)
    && record(settings) && strategy(settings.default_strategy)
    && Array.isArray(settings.k_values) && settings.k_values.every(item => integer(item, 1))
    && ratio(settings.min_similarity) && text(settings.distance) && integer(settings.chunk_size, 1)
    && integer(settings.chunk_overlap) && nullableInteger(settings.semantic_candidate_count, 1)
    && nullableInteger(settings.keyword_candidate_count, 1)
    && nullableInteger(settings.rrf_fusion_constant, 1) && Array.isArray(value.strategies)
    && value.strategies.length > 0 && value.strategies.every(ragStrategy)
    && Array.isArray(value.metric_definitions) && value.metric_definitions.every(item => record(item) && text(item.key) && text(item.label) && text(item.definition) && text(item.denominator))
    && text(value.comparability_note);
}
function agentMetric(value: unknown): value is AgentMetric {
  return record(value) && text(value.key) && text(value.label)
    && (value.value === null || ratio(value.value)) && text(value.denominator_name)
    && integer(value.denominator) && text(value.definition);
}
function agentSummary(value: unknown): value is AgentEvaluationSummary {
  if (!record(value)) return false;
  const dataset = value.dataset; const model = value.model; const settings = value.settings;
  const denominators = value.denominators; const activity = value.activity;
  const failures = value.failures; const usage = value.usage; const limitations = value.limitations;
  const observation = value.evidence_observation;
  return record(dataset) && text(dataset.version) && text(dataset.title) && text(dataset.license)
    && integer(dataset.document_count, 1) && integer(dataset.question_count, 1)
    && integer(dataset.repeats, 1) && integer(dataset.attempted_run_count, 1) && sha256(dataset.sha256)
    && run(value.run) && record(model) && text(model.provider) && text(model.model) && number(model.temperature)
    && record(settings) && text(settings.scope_prompt_version) && text(settings.decision_prompt_version)
    && text(settings.decision_schema_version) && text(settings.generation_schema_version)
    && integer(settings.graph_step_limit, 1) && integer(settings.max_tool_calls, 1)
    && integer(settings.tool_top_k, 1) && number(settings.execution_timeout_seconds, Number.MIN_VALUE)
    && strategy(settings.retrieval_strategy) && ratio(settings.min_similarity)
    && text(settings.embedding_model) && text(settings.embedding_revision)
    && record(denominators) && integer(denominators.attempted_runs, 1)
    && integer(denominators.completed_runs) && integer(denominators.failed_runs)
    && integer(denominators.expected_grounded_runs) && integer(denominators.expected_abstention_runs)
    && integer(denominators.citation_count) && integer(denominators.expected_relevant_reference_count)
    && integer(denominators.questions, 1) && Array.isArray(value.metrics) && value.metrics.every(agentMetric)
    && latency(value.latency) && record(activity) && distribution(activity.tool_calls)
    && distribution(activity.graph_steps) && (activity.stable_question_rate === null || ratio(activity.stable_question_rate))
    && integer(activity.questions_with_variation) && record(failures) && integer(failures.count)
    && ratio(failures.rate) && integer(failures.configured_timeout_overrun_count)
    && record(usage) && integer(usage.samples_with_usage) && integer(usage.prompt_tokens)
    && integer(usage.completion_tokens) && record(limitations) && text(limitations.citation_check)
    && limitations.factual_entailment_measured === false && text(limitations.sample_size)
    && (observation === null || (record(observation) && run(observation.run)
      && integer(observation.sample_size, 1) && text(observation.expected_status)
      && text(observation.observed_status) && integer(observation.retrieved_result_count)
      && integer(observation.evidence_count) && integer(observation.tool_call_count)
      && text(observation.finding) && text(observation.caution)))
    && text(value.comparability_note);
}

export function isEvaluationSummary(value: unknown): value is EvaluationSummary {
  return record(value) && value.schema_version === 1
    && (value.status === 'complete' || value.status === 'partial' || value.status === 'unavailable')
    && Array.isArray(value.artifacts) && value.artifacts.every(item => record(item)
      && (item.artifact_id === 'rag_baseline' || item.artifact_id === 'rag_strategy_reference'
        || item.artifact_id === 'agent_baseline' || item.artifact_id === 'agent_evidence_observation')
      && (item.state === 'loaded' || item.state === 'missing'
        || item.state === 'invalid' || item.state === 'incomplete' || item.state === 'incompatible'))
    && Array.isArray(value.issues) && value.issues.every(item => record(item)
      && text(item.artifact_id)
      && (item.code === 'artifact_missing' || item.code === 'artifact_invalid'
        || item.code === 'artifact_incomplete' || item.code === 'artifact_incompatible')
      && text(item.message))
    && (value.rag === null || ragSummary(value.rag))
    && (value.agentic === null || agentSummary(value.agentic))
    && Array.isArray(value.research_findings) && value.research_findings.every(item => record(item)
      && text(item.source) && text(item.title) && text(item.summary) && text(item.caution))
    && value.cross_dataset_comparison_allowed === false
    && value.generated_from_versioned_artifacts_only === true;
}

export async function requestEvaluationSummary(
  signal: AbortSignal,
  fetcher: FetchLike = fetch,
): Promise<EvaluationSummary> {
  const response = await fetcher('/api/evaluations/summary', { signal });
  let body: unknown;
  try { body = await response.json(); }
  catch { throw new Error(`The evaluation API returned unreadable data (HTTP ${response.status}).`); }
  if (!response.ok) throw new Error(`Evaluation summaries are unavailable (HTTP ${response.status}).`);
  if (!isEvaluationSummary(body)) throw new Error('The evaluation API returned an incompatible summary format.');
  return body;
}
