import { useEffect, useState } from 'react';

import {
  requestEvaluationSummary,
  type AgentEvaluationSummary, type EvaluationSummary, type RagEvaluationSummary,
} from './EvaluationContract.js';

function percent(value: number | null): string {
  return value === null ? 'Unavailable' : `${(value * 100).toFixed(1)}%`;
}
function duration(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(2)} ms`;
}
function date(value: string): string { return value.slice(0, 10); }
function label(value: string): string { return value.replaceAll('_', ' '); }

function MetricBar({ label: name, value, detail }: { label: string; value: number | null; detail: string }) {
  const width = value === null ? 0 : Math.max(0, Math.min(100, value * 100));
  return <div className="evaluation-bar-row">
    <div><strong>{name}</strong><span>{detail}</span></div>
    <div className="evaluation-bar-track" role="img" aria-label={`${name}: ${percent(value)}; ${detail}`}>
      <span style={{ width: `${width}%` }} />
    </div>
    <b>{percent(value)}</b>
  </div>;
}

function RagEvaluation({ summary }: { summary: RagEvaluationSummary }) {
  return <article className="evaluation-panel evaluation-rag" aria-labelledby="evaluation-rag-title">
    <div className="evaluation-panel-heading"><div><span>RETRIEVAL EVALUATION</span><h3 id="evaluation-rag-title">RAG evidence retrieval</h3></div><small>{summary.dataset.version}</small></div>
    <p className="evaluation-context">{summary.dataset.question_count} questions: {summary.dataset.answerable_question_count} answerable / {summary.dataset.unanswerable_question_count} unanswerable. Metrics below macro-average only the answerable questions.</p>
    <div className="evaluation-model"><strong>{summary.model.model}</strong><span>{summary.model.dimension} dimensions / {summary.model.normalization} / revision {summary.model.revision.slice(0, 12)}</span><span>Baseline {date(summary.baseline_run.created_at_utc)}{summary.strategy_reference_run && <> / strategy reference {date(summary.strategy_reference_run.created_at_utc)}</>}</span></div>
    <div className="evaluation-strategies">{summary.strategies.map(strategy => <section key={strategy.strategy} className="evaluation-strategy">
      <div className="evaluation-strategy-heading"><h4>{label(strategy.strategy)}</h4>{strategy.strategy === summary.settings.default_strategy && <span>DEFAULT</span>}</div>
      <p>Latency ({strategy.latency.sample_count} queries): p50 {duration(strategy.latency.p50_ms)} / p95 {duration(strategy.latency.p95_ms)}</p>
      <div className="evaluation-table-scroll"><table><caption className="sr-only">{label(strategy.strategy)} retrieval metrics</caption><thead><tr><th>K</th><th>Recall</th><th>Precision</th><th>MRR</th><th>Hit rate</th></tr></thead><tbody>{strategy.metrics.map(metric => <tr key={metric.k}><th>{metric.k}</th><td>{percent(metric.recall_at_k)}</td><td>{percent(metric.precision_at_k)}</td><td>{percent(metric.mrr_at_k)}</td><td>{percent(metric.hit_rate_at_k)}</td></tr>)}</tbody></table></div>
      {strategy.metrics.map(metric => <MetricBar key={metric.k} label={`Recall@${metric.k}`} value={metric.recall_at_k} detail={`${metric.answerable_question_count} answerable questions`} />)}
    </section>)}</div>
    <details className="evaluation-definitions"><summary>RAG settings and metric definitions</summary><p>Exact cosine / minimum similarity {summary.settings.min_similarity} / chunks {summary.settings.chunk_size} with {summary.settings.chunk_overlap} overlap.</p><dl>{summary.metric_definitions.map(item => <div key={item.key}><dt>{item.label}</dt><dd>{item.definition} Denominator: {item.denominator}.</dd></div>)}</dl></details>
    <p className="evaluation-comparability">{summary.comparability_note}</p>
  </article>;
}

function AgentEvaluation({ summary }: { summary: AgentEvaluationSummary }) {
  return <article className="evaluation-panel evaluation-agent" aria-labelledby="evaluation-agent-title">
    <div className="evaluation-panel-heading"><div><span>DECISION &amp; RELIABILITY EVALUATION</span><h3 id="evaluation-agent-title">Agentic behavior</h3></div><small>{summary.dataset.version}</small></div>
    <p className="evaluation-context">{summary.dataset.question_count} questions × {summary.dataset.repeats} repeats = {summary.denominators.attempted_runs} attempted runs. Failures remain in attempted-run accuracy denominators.</p>
    <div className="evaluation-model"><strong>{summary.model.provider} / {summary.model.model}</strong><span>Temperature {summary.model.temperature} / {summary.settings.retrieval_strategy} retrieval / {summary.settings.execution_timeout_seconds}s deadline</span><span>Baseline {date(summary.run.created_at_utc)}</span></div>
    <div className="evaluation-bars">{summary.metrics.map(metric => <MetricBar key={metric.key} label={metric.label} value={metric.value} detail={`${metric.denominator} ${metric.denominator_name}`} />)}</div>
    <div className="evaluation-operational" aria-label="Agentic operational metrics">
      <div><span>Latency p50</span><strong>{duration(summary.latency.p50_ms)}</strong><small>{summary.latency.sample_count} attempts</small></div>
      <div><span>Latency p95</span><strong>{duration(summary.latency.p95_ms)}</strong><small>includes failed attempts</small></div>
      <div><span>Failure rate</span><strong>{percent(summary.failures.rate)}</strong><small>{summary.failures.count} / {summary.denominators.attempted_runs}</small></div>
      <div><span>Mean tool calls</span><strong>{summary.activity.tool_calls.mean.toFixed(2)}</strong><small>{summary.activity.tool_calls.sample_count} completed runs</small></div>
      <div><span>Mean graph steps</span><strong>{summary.activity.graph_steps.mean.toFixed(2)}</strong><small>{summary.activity.graph_steps.sample_count} completed runs</small></div>
      <div><span>Stable questions</span><strong>{percent(summary.activity.stable_question_rate)}</strong><small>{summary.activity.questions_with_variation} varied</small></div>
    </div>
    <div className="evaluation-denominators"><h4>Denominators remain separate</h4><dl><div><dt>Completed / failed</dt><dd>{summary.denominators.completed_runs} / {summary.denominators.failed_runs}</dd></div><div><dt>Grounded-labelled runs</dt><dd>{summary.denominators.expected_grounded_runs}</dd></div><div><dt>Abstention-labelled runs</dt><dd>{summary.denominators.expected_abstention_runs}</dd></div><div><dt>Returned citations</dt><dd>{summary.denominators.citation_count}</dd></div><div><dt>Expected references</dt><dd>{summary.denominators.expected_relevant_reference_count}</dd></div></dl></div>
    {summary.evidence_observation && <aside className="evaluation-observation"><h4>Targeted evidence-assessment observation</h4><p>{summary.evidence_observation.finding}</p><p>{summary.evidence_observation.retrieved_result_count} results / {summary.evidence_observation.evidence_count} retained chunks / {summary.evidence_observation.tool_call_count} tool call / expected {label(summary.evidence_observation.expected_status)} / observed {label(summary.evidence_observation.observed_status)}.</p><small>{summary.evidence_observation.caution}</small></aside>}
    <details className="evaluation-definitions"><summary>Agent settings, definitions and limitations</summary><p>Scope prompt {summary.settings.scope_prompt_version} / decision prompt {summary.settings.decision_prompt_version} / decision schema {summary.settings.decision_schema_version} / max {summary.settings.max_tool_calls} tools / {summary.settings.graph_step_limit} graph steps.</p><dl>{summary.metrics.map(item => <div key={item.key}><dt>{item.label}</dt><dd>{item.definition}</dd></div>)}</dl><p>{summary.limitations.citation_check}. Factual entailment measured: no. {summary.limitations.sample_size}.</p></details>
    <p className="evaluation-comparability">{summary.comparability_note}</p>
  </article>;
}

export function EvaluationLoading() {
  return <div className="evaluation-state" role="status">Loading versioned evaluation summaries...</div>;
}
export function EvaluationError({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return <div className="evaluation-state evaluation-state-error" role="alert"><strong>Evaluation summaries could not be loaded.</strong><p>{message}</p>{onRetry && <button type="button" onClick={onRetry}>Retry summaries</button>}</div>;
}
export function EvaluationEmpty() {
  return <div className="evaluation-state" role="status"><strong>No compatible evaluation summary is available.</strong><p>Review the artifact-status details for missing, incomplete or incompatible versioned files.</p></div>;
}

export function EvaluationDashboardContent({ summary }: { summary: EvaluationSummary }) {
  const empty = !summary.rag && !summary.agentic;
  return <>
    {summary.issues.length > 0 && <div className="evaluation-issues" role="status"><strong>Artifact status: {summary.status}</strong><ul>{summary.issues.map(item => <li key={`${item.artifact_id}-${item.code}`}><code>{item.artifact_id}</code>: {item.message}</li>)}</ul></div>}
    {empty ? <EvaluationEmpty /> : <div className="evaluation-grid">
      {summary.rag && <RagEvaluation summary={summary.rag} />}
      {summary.agentic && <AgentEvaluation summary={summary.agentic} />}
    </div>}
    {!!summary.research_findings.length && <section className="research-findings" aria-labelledby="research-findings-title"><h3 id="research-findings-title">Research findings</h3><p>These panels summarize different datasets and tasks. They do not establish a universal winner.</p><div>{summary.research_findings.map(item => <article key={item.source}><h4>{item.title}</h4><p>{item.summary}</p><small>{item.caution}</small></article>)}</div><aside><strong>Live comparison observation:</strong> the documented Germany PDF run produced a validated RAG citation while Agentic retrieved relevant evidence and abstained. This is consistent with an evidence-assessment limitation, not proof that one architecture is universally better.</aside></section>}
    <details className="evaluation-artifacts"><summary>Versioned artifact status</summary><ul>{summary.artifacts.map(item => <li key={item.artifact_id}><code>{item.artifact_id}</code><span>{item.state}</span></li>)}</ul></details>
  </>;
}

export function EvaluationDashboard() {
  const [summary, setSummary] = useState<EvaluationSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController(); setLoading(true); setError('');
    requestEvaluationSummary(controller.signal).then(value => {
      if (!controller.signal.aborted) setSummary(value);
    }).catch(reason => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Evaluation summaries are unavailable.');
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [retry]);
  return <section className="evaluation-dashboard" aria-labelledby="evaluation-dashboard-title" aria-busy={loading}>
    <div className="evaluation-dashboard-heading"><div><span className="comparison-pipeline">VERSIONED OFFLINE EVIDENCE</span><h2 id="evaluation-dashboard-title">Evaluation dashboard</h2><p>Read-only summaries of committed RAG and Agentic artifacts. No evaluation or inference runs when this section loads.</p></div>{summary && <span className={`evaluation-status status-${summary.status}`}>{summary.status}</span>}</div>
    <div className="comparison-quality-note"><strong>Not directly comparable.</strong> RAG metrics evaluate retrieval rankings; Agentic metrics evaluate routing, status, grounding and reliability on another dataset with separate denominators.</div>
    {loading ? <EvaluationLoading /> : error ? <EvaluationError message={error} onRetry={() => setRetry(value => value + 1)} /> : summary ? <EvaluationDashboardContent summary={summary} /> : <EvaluationEmpty />}
  </section>;
}
