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
