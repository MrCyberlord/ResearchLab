import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

import {
  isComparisonResponse,
  submitComparison,
} from '../.cache/compare-tests/CompareContract.js';
import {
  ComparisonProgress,
  ComparisonResults,
} from '../.cache/compare-tests/CompareQuestion.js';
import {
  isEvaluationSummary,
  requestEvaluationSummary,
} from '../.cache/compare-tests/EvaluationContract.js';
import {
  EvaluationDashboardContent, EvaluationEmpty, EvaluationError, EvaluationLoading,
} from '../.cache/compare-tests/EvaluationDashboard.js';

const citation = {
  source_id: 'S1', chunk_id: 49, document_id: '03baf4c4-fe58-4b74-8b31-34995b329c6a',
  filename: 'germany_country_profile.pdf', page_number: 1, chunk_order: 0,
  quote: 'Population\n83.47 million (31 December 2025)',
};

const trace = {
  node: 'decide', duration_ms: 12.5, outcome: 'answer_grounded',
  decision_action: 'answer_grounded', decision_scope: 'document_research',
  decision_reason: 'evidence_supports_answer', evidence_count: 1,
  remaining_budget_ms: 90000, error_type: null, error_code: null,
  validation_retry_count: null, validation_failure_code: null,
};

const tool = {
  call_index: 1, top_k: 3, result_count: 1, duration_ms: 8,
  outcome: 'results', query_fingerprint: '0123456789abcdef',
  retrieval_strategy: 'semantic', document_filter_count: 1,
  evidence_added_count: 1, remaining_budget_ms: 100000,
};

function successfulResponse() {
  return {
    execution_mode: 'sequential', total_elapsed_ms: 1200,
    document_scope: 'selected_documents', requested_strategy: 'semantic',
    activity_metrics_are_quality_measures: false,
    rag: {
      pipeline: 'rag', status: 'answered', grounding_mode: 'document_grounded',
      document_scope: 'selected_documents', answer: "Germany's population is 83.47 million [S1].",
      message: null, citations: [citation], elapsed_ms: 500, error: null,
      provider: 'ollama', model: 'gemma3:4b', retrieval_strategy: 'semantic',
      retrieval_ms: 100, generation_ms: 400, retrieved_chunk_count: 3,
      context_chunk_count: 3, usage: { prompt_tokens: 100, completion_tokens: 20 },
    },
    agentic: {
      pipeline: 'agentic', status: 'grounded', grounding_mode: 'document_grounded',
      document_scope: 'selected_documents', route: 'document_research',
      answer: 'The document reports 83.47 million [S1].', message: null,
      citations: [citation], elapsed_ms: 700, error: null, provider: 'ollama', model: 'gemma3:4b',
      graph_steps: 1, tool_call_count: 1, node_trace: [trace], tool_activity: [tool],
      usage: { prompt_tokens: 140, completion_tokens: 25 }, execution_budget_ms: 120000,
      remaining_budget_ms: 90000,
    },
  };
}

function evaluationResponse() {
  return {
    schema_version: 1, status: 'complete',
    artifacts: ['rag_baseline', 'rag_strategy_reference', 'agent_baseline', 'agent_evidence_observation'].map(artifact_id => ({ artifact_id, state: 'loaded' })),
    issues: [], cross_dataset_comparison_allowed: false,
    generated_from_versioned_artifacts_only: true,
    rag: {
      dataset: { version: 'researchlab-rag-eval-v1', title: 'Synthetic habitat operations corpus', license: 'CC0-1.0', document_count: 4, question_count: 8, answerable_question_count: 6, unanswerable_question_count: 2, sha256: 'a'.repeat(64) },
      baseline_run: { run_id: 'rag-v1', created_at_utc: '2026-10-29T05:28:55+00:00' },
      strategy_reference_run: { run_id: 'rag-strategies-v1', created_at_utc: '2026-10-29T06:08:46+00:00' },
      model: { provider: 'sentence_transformers', model: 'sentence-transformers/all-MiniLM-L6-v2', revision: '1110a243fdf4706b3f48f1d95db1a4f5529b4d41', dimension: 384, normalization: 'L2 normalized' },
      settings: { default_strategy: 'semantic', k_values: [3], min_similarity: 0.15, distance: 'pgvector exact cosine', chunk_size: 700, chunk_overlap: 100, semantic_candidate_count: 20, keyword_candidate_count: 20, rrf_fusion_constant: 60 },
      strategies: [{ strategy: 'semantic', reference_only: true, metrics: [{ k: 3, recall_at_k: 1, precision_at_k: 0.388889, mrr_at_k: 0.916667, hit_rate_at_k: 1, answerable_question_count: 6 }], latency: { sample_count: 8, p50_ms: 17.03, p95_ms: 21.57 } }],
      metric_definitions: [{ key: 'recall_at_k', label: 'Recall@K', definition: 'Relevant references retrieved divided by all relevant references.', denominator: '6 answerable questions' }],
      comparability_note: 'RAG retrieval metrics are not directly comparable to Agentic decision metrics.',
    },
    agentic: {
      dataset: { version: 'researchlab-agent-eval-v1', title: 'Synthetic document-research agent evaluation', license: 'CC0-1.0', document_count: 3, question_count: 10, repeats: 2, attempted_run_count: 20, sha256: 'b'.repeat(64) },
      run: { run_id: 'agent-v1', created_at_utc: '2026-10-30T13:44:18+00:00' },
      model: { provider: 'ollama', model: 'gemma3:4b', temperature: 0 },
      settings: { scope_prompt_version: 'scope-v2', decision_prompt_version: 'document-action-v2', decision_schema_version: 'document-action-schema-v2', generation_schema_version: 'agent-answer-schema-v1', graph_step_limit: 20, max_tool_calls: 3, tool_top_k: 5, execution_timeout_seconds: 120, retrieval_strategy: 'semantic', min_similarity: 0.15, embedding_model: 'sentence-transformers/all-MiniLM-L6-v2', embedding_revision: '1110a243fdf4706b3f48f1d95db1a4f5529b4d41' },
      denominators: { attempted_runs: 20, completed_runs: 17, failed_runs: 3, expected_grounded_runs: 8, expected_abstention_runs: 6, citation_count: 3, expected_relevant_reference_count: 8, questions: 10 },
      metrics: [{ key: 'tool_selection_accuracy', label: 'Tool-selection accuracy', value: 0.85, denominator_name: 'attempted runs', denominator: 20, definition: 'Expected tool use matched; errors are incorrect.' }, { key: 'citation_validity', label: 'Citation validity', value: 1, denominator_name: 'returned citations', denominator: 3, definition: 'Mechanical source identity and quote containment.' }],
      latency: { sample_count: 20, p50_ms: 19443.13, p95_ms: 40652.15 },
      activity: { tool_calls: { sample_count: 17, mean: 0.647, minimum: 0, maximum: 1 }, graph_steps: { sample_count: 17, mean: 4.941, minimum: 3, maximum: 6 }, stable_question_rate: 0.9, questions_with_variation: 1 },
      failures: { count: 3, rate: 0.15, configured_timeout_overrun_count: 1 },
      usage: { samples_with_usage: 17, prompt_tokens: 10808, completion_tokens: 1236 },
      limitations: { citation_check: 'Mechanical checks only', factual_entailment_measured: false, sample_size: 'Small synthetic dataset' },
      evidence_observation: { run: { run_id: 'agent-observation-v1', created_at_utc: '2026-10-30T17:09:46+00:00' }, sample_size: 1, expected_status: 'grounded', observed_status: 'insufficient_evidence', retrieved_result_count: 2, evidence_count: 2, tool_call_count: 1, finding: 'The targeted run retrieved evidence but abstained.', caution: 'One run is not a general estimate.' },
      comparability_note: 'Agentic metrics use another task and denominator.',
    },
    research_findings: [{ source: 'rag_baseline', title: 'RAG retrieval reference', summary: 'Recall@3 reached 100% on six answerable questions.', caution: 'Small synthetic dataset.' }, { source: 'agent_evidence_observation', title: 'Evidence assessment', summary: 'Evidence was retrieved before abstention.', caution: 'No universal winner.' }],
  };
}

async function mockedComparison(response) {
  let callCount = 0;
  const fetcher = async () => {
    callCount += 1;
    return new Response(JSON.stringify(response), {
      status: 200, headers: { 'Content-Type': 'application/json' },
    });
  };
  const result = await submitComparison({
    question: "What is Germany's population?",
    document_ids: [citation.document_id], strategy: 'semantic',
  }, new AbortController().signal, fetcher);
  assert.equal(callCount, 1);
  return result;
}

test('one API helper call targets only the comparison endpoint', async () => {
  const calls = [];
  const response = successfulResponse();
  const fetcher = async (url, options) => {
    calls.push({ url, options });
    return new Response(JSON.stringify(response), {
      status: 200, headers: { 'Content-Type': 'application/json' },
    });
  };
  const controller = new AbortController();
  const result = await submitComparison({
    question: "What is Germany's population?",
    document_ids: [citation.document_id], strategy: 'semantic',
  }, controller.signal, fetcher);
  assert.equal(result.execution_mode, 'sequential');
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/api/compare/questions');
  assert.equal(calls[0].options.method, 'POST');
  assert.equal(calls[0].options.signal, controller.signal);
  assert.deepEqual(JSON.parse(calls[0].options.body), {
    question: "What is Germany's population?",
    document_ids: [citation.document_id], strategy: 'semantic',
  });
});

test('success renders independent answers, validated citations and operational metrics', async () => {
  const result = await mockedComparison(successfulResponse());
  assert.equal(isComparisonResponse(result), true);
  const html = renderToStaticMarkup(createElement(ComparisonResults, { result }));
  assert.match(html, /RAG/);
  assert.match(html, /Agentic/);
  assert.match(html, /83\.47 million/);
  assert.match(html, /germany_country_profile\.pdf/);
  assert.match(html, /page 1/);
  assert.match(html, /Population\n83\.47 million/);
  assert.match(html, /href="#compare-rag-source-S1"/);
  assert.match(html, /500 ms/);
  assert.match(html, /700 ms/);
  assert.match(html, /Chunks retrieved/);
  assert.match(html, /Graph steps/);
  assert.match(html, /Tool calls/);
  assert.match(html, /semantic/);
  assert.match(html, /No automatic winner/);
  assert.match(html, /not answer-quality scores/);
});

test('mocked Agentic abstention remains visible beside a successful RAG result', async () => {
  const response = successfulResponse();
  response.agentic = {
    ...response.agentic, status: 'insufficient_evidence',
    grounding_mode: 'insufficient_document_evidence', answer: null,
    message: 'The selected documents did not provide enough evidence.', citations: [],
  };
  const result = await mockedComparison(response);
  assert.equal(isComparisonResponse(result), true);
  const html = renderToStaticMarkup(createElement(ComparisonResults, { result }));
  assert.match(html, /Germany&#x27;s population is 83\.47 million/);
  assert.match(html, /Insufficient evidence/);
  assert.match(html, /did not provide enough evidence/);
});

test('mocked partial pipeline error does not hide the other pipeline answer', async () => {
  const response = successfulResponse();
  response.rag = {
    ...response.rag, status: 'error', answer: null, citations: [],
    error: { code: 'pipeline_timeout', category: 'timeout', message: 'The bounded pipeline timed out.', retryable: true },
    retrieval_strategy: null, retrieval_ms: null, generation_ms: null,
    retrieved_chunk_count: null, context_chunk_count: null, usage: null,
  };
  response.agentic = {
    ...response.agentic, status: 'direct', grounding_mode: 'general_knowledge', route: 'direct',
    answer: 'Germany is a country in Central Europe.', citations: [], tool_call_count: 0,
    tool_activity: [],
  };
  const result = await mockedComparison(response);
  assert.equal(isComparisonResponse(result), true);
  const html = renderToStaticMarkup(createElement(ComparisonResults, { result }));
  assert.match(html, /Pipeline error/);
  assert.match(html, /pipeline_timeout/);
  assert.match(html, /Germany is a country in Central Europe/);
});

test('Agentic tool activity and safe node traces are expandable', async () => {
  const result = await mockedComparison(successfulResponse());
  const html = renderToStaticMarkup(createElement(ComparisonResults, { result }));
  assert.match(html, /<details class="comparison-details">/);
  assert.match(html, /Tool activity \(1\)/);
  assert.match(html, /Safe execution trace \(1 nodes\)/);
  assert.match(html, /0123456789abcdef/);
  assert.match(html, /evidence supports answer/);
  assert.doesNotMatch(html, /raw search query/i);
});

test('long-running loading state explains sequence and lack of a short timeout', () => {
  const html = renderToStaticMarkup(createElement(ComparisonProgress));
  assert.match(html, /Running both pipelines sequentially/);
  assert.match(html, /RAG runs first, followed by Agentic/);
  assert.match(html, /does not impose a short client timeout/);
});

test('component has a synchronous duplicate guard and disables submit while busy', () => {
  const source = readFileSync(new URL('../src/CompareQuestion.tsx', import.meta.url), 'utf8');
  const submitStart = source.indexOf('async function compare(');
  const guard = source.indexOf('if (comparisonRequest.current) return;', submitStart);
  const requestStart = source.indexOf('comparisonRequest.current = controller;', submitStart);
  assert.ok(submitStart >= 0 && guard > submitStart);
  assert.ok(requestStart > guard, 'the in-flight guard must run before a new request is stored');
  assert.match(source, /<button type="submit" disabled=\{busy\}>/);
});

test('comparison layout is side by side and has an explicit mobile stack rule', () => {
  const css = readFileSync(new URL('../src/compare.css', import.meta.url), 'utf8');
  assert.match(css, /\.comparison-grid\{display:grid;grid-template-columns:minmax\(0,1fr\) minmax\(0,1fr\)/);
  assert.match(css, /@media\(max-width:900px\)\{\.comparison-grid\{grid-template-columns:minmax\(0,1fr\)\}/);
});

test('evaluation API helper loads only the aggregate summary endpoint', async () => {
  const calls = [];
  const fetcher = async (url, options) => {
    calls.push({ url, options });
    return new Response(JSON.stringify(evaluationResponse()), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };
  const controller = new AbortController();
  const result = await requestEvaluationSummary(controller.signal, fetcher);
  assert.equal(result.status, 'complete');
  assert.deepEqual(calls.map(item => item.url), ['/api/evaluations/summary']);
  assert.equal(calls[0].options.signal, controller.signal);
});

test('evaluation dashboard renders separate accessible RAG and Agentic evidence', () => {
  const summary = evaluationResponse();
  assert.equal(isEvaluationSummary(summary), true);
  const html = renderToStaticMarkup(createElement(EvaluationDashboardContent, { summary }));
  assert.match(html, /RAG evidence retrieval/);
  assert.match(html, /Agentic behavior/);
  assert.match(html, /aria-label="Recall@3: 100\.0%; 6 answerable questions"/);
  assert.match(html, /19\.44 s/);
  assert.match(html, /Failure rate/);
  assert.match(html, /15\.0%/);
  assert.match(html, /Mean tool calls/);
  assert.match(html, /Completed \/ failed/);
  assert.match(html, /17 \/ 3/);
  assert.match(html, /Targeted evidence-assessment observation/);
  assert.match(html, /retrieved evidence but abstained/);
  assert.match(html, /do not establish a universal winner/);
});

test('evaluation loading, empty and error states are readable', () => {
  const loading = renderToStaticMarkup(createElement(EvaluationLoading));
  const empty = renderToStaticMarkup(createElement(EvaluationEmpty));
  const error = renderToStaticMarkup(createElement(EvaluationError, { message: 'Safe summary unavailable.' }));
  assert.match(loading, /Loading versioned evaluation summaries/);
  assert.match(empty, /No compatible evaluation summary is available/);
  assert.match(error, /Evaluation summaries could not be loaded/);
  assert.match(error, /Safe summary unavailable/);
});

test('evaluation contract rejects incomplete aggregate data', () => {
  const summary = evaluationResponse();
  delete summary.agentic.denominators.citation_count;
  assert.equal(isEvaluationSummary(summary), false);
  const incompatible = evaluationResponse();
  incompatible.agentic.settings.retrieval_strategy = 'keyword_only';
  assert.equal(isEvaluationSummary(incompatible), false);
});
