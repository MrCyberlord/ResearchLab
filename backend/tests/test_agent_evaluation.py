"""Deterministic agent-evaluation tests; no real model or PostgreSQL required."""
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest

from app.domain.agent import (
    AgentExecutionLimit, AgentExecutionMetadata, AgentLLMInvalidResponse,
    AgentLLMTimeout, AgentNodeTrace, AgentResult, AgentTokenUsage, AgentToolActivity,
)
from app.domain.rag import Citation
from app.evaluation.agent_cli import report
from app.evaluation.agent_dataset import default_agent_dataset_path, load_agent_dataset
from app.evaluation.agent_runner import (
    AgentEvaluationRunner, agent_metrics, compare_agent_results, decision_variability,
)
from app.evaluation.runner import EvaluationEnvironment


def test_versioned_agent_dataset_is_licensed_and_covers_required_scenarios():
    dataset = load_agent_dataset(default_agent_dataset_path())
    tags = {tag for question in dataset.questions for tag in question.tags}
    assert dataset.version == "researchlab-agent-eval-v1"
    assert dataset.license == "CC0-1.0"
    assert len(dataset.documents) == 3 and len(dataset.questions) == 10
    assert {
        "general_knowledge", "document_grounded", "unanswerable_document",
        "paraphrase", "ambiguous", "repeated_search",
    } <= tags
    assert sum(question.expected_status == "direct" for question in dataset.questions) == 3
    assert sum(question.expected_status == "grounded" for question in dataset.questions) == 4
    assert sum(question.expected_status == "insufficient_evidence"
               for question in dataset.questions) == 3
    assert (default_agent_dataset_path().parent / "LICENSE.md").is_file()
    baseline = default_agent_dataset_path().parents[2] / "baselines" / "gemma3-4b-v1.json"
    baseline_result = json.loads(baseline.read_text("utf-8"))
    assert baseline_result["dataset"]["sha256"] == dataset.checksum
    assert baseline_result["dataset"]["repeats"] == 2


def test_agent_metrics_use_explicit_denominators_and_count_errors_as_incorrect():
    reference = ("doc", 1, 0)
    cases = [
        {"expected_scope": "general_knowledge", "actual_scope": "general_knowledge",
         "expected_status": "direct", "status": "direct", "expected_tool_use": False,
         "tool_used": False, "tool_call_count": 0, "graph_steps": 3, "latency_ms": 10,
         "relevant": [], "citations": []},
        {"expected_scope": "document_research", "actual_scope": "document_research",
         "expected_status": "grounded", "status": "grounded", "expected_tool_use": True,
         "tool_used": True, "tool_call_count": 1, "graph_steps": 6, "latency_ms": 30,
         "relevant": [reference], "citations": [{"reference_key": reference,
                                                    "reference_match": True,
                                                    "quote_valid": True}]},
        {"expected_scope": "document_research", "actual_scope": "document_research",
         "expected_status": "insufficient_evidence", "status": "insufficient_evidence",
         "expected_tool_use": True, "tool_used": True, "tool_call_count": 1,
         "graph_steps": 6, "latency_ms": 20, "relevant": [], "citations": []},
        {"expected_scope": "general_knowledge", "actual_scope": None,
         "expected_status": "direct", "status": "error", "expected_tool_use": False,
         "tool_used": None, "tool_call_count": None, "graph_steps": None,
         "latency_ms": 40, "relevant": [], "citations": [], "error_type": "Timeout"},
    ]
    result = agent_metrics(cases)
    assert result["tool_selection_accuracy"] == .75
    assert result["document_scope_compliance"] == .75
    assert result["expected_status_accuracy"] == .75
    assert result["grounded_answer_rate"] == 1
    assert result["abstention_accuracy"] == .75
    assert result["unanswerable_abstention_rate"] == 1
    assert result["citation_validity"] == 1 and result["citation_coverage"] == 1
    assert result["end_to_end_latency_ms"] == {"sample_count": 4, "p50": 20, "p95": 40}
    assert result["denominators"] == {
        "attempted_runs": 4, "completed_runs": 3, "expected_grounded_runs": 1,
        "expected_abstention_runs": 1, "citation_count": 1,
        "expected_relevant_reference_count": 1,
    }
    assert result["factual_entailment_measured"] is False


class FakeAgentService:
    def __init__(self, dataset, environment, *, vary=False, failure=None):
        self.dataset = dataset
        self.environment = environment
        self.vary = vary
        self.failure = failure
        self.calls = []
        self.counts = {}
        self.graph_step_limit = 20
        self.execution_timeout_seconds = 120
        self.graph = SimpleNamespace(
            provider=SimpleNamespace(provider="fake", model="fake-model"),
            max_tool_calls=3, max_tool_top_k=5,
        )

    def run(self, question_text, document_ids, strategy, include_metadata=True):
        self.calls.append((question_text, document_ids, strategy, include_metadata))
        if self.failure:
            raise self.failure
        question = next(item for item in self.dataset.questions if item.question == question_text)
        self.counts[question.id] = self.counts.get(question.id, 0) + 1
        status = question.expected_status
        route = "direct" if question.expected_scope == "general_knowledge" else "document_research"
        tool_count = 1 if question.expected_tool_use else 0
        if self.vary and question.id == "g01_llm_definition" and self.counts[question.id] == 2:
            status, route, tool_count = "insufficient_evidence", "document_research", 1
        citations = []
        if status == "grounded":
            reference = question.relevant[0]
            document_id = self.environment.document_ids[reference.document]
            citations = [Citation(
                "S1", 1, document_id, reference.document + ".txt", reference.page,
                reference.chunk_order, "synthetic quote", "synthetic quote and context",
            )]
        activities = ([AgentToolActivity(1, "synthetic query", 3, 1 if citations else 0,
                                         .1, "results" if citations else "empty")]
                      if tool_count else [])
        metadata = AgentExecutionMetadata(
            route, 1, 3 + 3 * tool_count, tool_count,
            [AgentNodeTrace("decide", .1, status)], activities,
            AgentTokenUsage(10, 5),
        )
        return AgentResult(
            status, "answer" if status != "insufficient_evidence" else None,
            "insufficient" if status == "insufficient_evidence" else None,
            citations, "fake", "fake-model", True, metadata,
        )


def environment_for(dataset):
    ids = {document.key: uuid5(NAMESPACE_URL, document.key) for document in dataset.documents}
    retrieval = SimpleNamespace(embeddings=SimpleNamespace(
        model="fake-embedding", revision="r1", profile="p1",
    ))
    return EvaluationEnvironment(
        retrieval, None, ids, {value: key for key, value in ids.items()}, "test",
    )


def settings():
    return SimpleNamespace()


def test_runner_records_repeat_variability_filters_and_machine_readable_cases(tmp_path):
    dataset = load_agent_dataset(default_agent_dataset_path())
    environment = environment_for(dataset)
    service = FakeAgentService(dataset, environment, vary=True)
    result = AgentEvaluationRunner(
        dataset, environment, settings(), "fixed", tmp_path, service,
    ).run(repeats=2, strategy="semantic")
    assert result["dataset"]["attempted_run_count"] == 20
    assert result["metrics"]["error_count"] == 0
    assert result["metrics"]["configured_timeout_overrun_count"] == 0
    assert result["variability"]["questions_with_variation"] == 1
    assert result["variability"]["stable_question_rate"] == .9
    filtered = next(question for question in dataset.questions if question.document_filter)
    filtered_calls = [call for call in service.calls if call[0] == filtered.question]
    assert all(call[1] == [environment.document_ids[filtered.document_filter[0]]]
               for call in filtered_calls)
    assert all(call[2:] == ("semantic", True) for call in service.calls)
    json.dumps(result)
    human = report(result)
    assert "does not measure factual entailment" in human
    assert "20 attempted runs" in human


@pytest.mark.parametrize("failure", [
    AgentLLMInvalidResponse("malformed"), AgentLLMTimeout("timeout"),
    AgentExecutionLimit("limit"),
])
def test_runner_failure_injection_is_visible_and_never_scores_as_success(tmp_path, failure):
    dataset = load_agent_dataset(default_agent_dataset_path())
    environment = environment_for(dataset)
    service = FakeAgentService(dataset, environment, failure=failure)
    result = AgentEvaluationRunner(
        dataset, environment, settings(), "failure", tmp_path, service,
    ).run()
    assert result["metrics"]["error_count"] == len(dataset.questions)
    assert result["metrics"]["tool_selection_accuracy"] == 0
    assert result["metrics"]["document_scope_compliance"] == 0
    assert {case["error_type"] for case in result["cases"]} == {type(failure).__name__}


def test_runner_preserves_safe_partial_trace_on_execution_failure(tmp_path):
    dataset = load_agent_dataset(default_agent_dataset_path())
    environment = environment_for(dataset)
    failure = AgentLLMInvalidResponse("malformed private response")
    failure.agent_trace = [AgentNodeTrace(
        "grounded_response", 2.5, "failed", "answer_grounded",
        "document_research", "evidence_supports_answer", 2, 5000,
        "LLMValidationFailure", "quote_mismatch", 1, "quote_mismatch",
    )]
    failure.agent_tool_activity = [AgentToolActivity(
        1, "private model query", 3, 2, 1.2, "results",
        "0123456789abcdef", "semantic", 1, 2, 6000,
    )]
    failure.agent_route = "document_research"
    failure.agent_tool_call_count = 1
    failure.agent_remaining_budget_ms = 5000
    failure.agent_error_code = "quote_mismatch"
    result = AgentEvaluationRunner(
        dataset, environment, settings(), "partial", tmp_path,
        FakeAgentService(dataset, environment, failure=failure),
    ).run(question_ids=["d03_low_battery_response"])
    case = result["cases"][0]
    assert case["error_type"] == "AgentLLMInvalidResponse"
    assert case["error_code"] == "quote_mismatch"
    assert case["actual_scope"] == "document_research"
    assert case["tool_used"] is True and case["tool_call_count"] == 1
    assert case["node_trace"][0]["validation_retry_count"] == 1
    assert case["remaining_budget_ms"] == 5000


def test_runner_can_limit_an_expensive_real_model_run_to_known_questions(tmp_path):
    dataset = load_agent_dataset(default_agent_dataset_path())
    environment = environment_for(dataset)
    service = FakeAgentService(dataset, environment)
    result = AgentEvaluationRunner(
        dataset, environment, settings(), "targeted", tmp_path, service,
    ).run(question_ids=["d01_seedling_protection", "d02_seedling_paraphrase"])
    assert result["dataset"]["question_count"] == 2
    assert result["dataset"]["total_question_count"] == 10
    assert result["dataset"]["selected_question_ids"] == [
        "d01_seedling_protection", "d02_seedling_paraphrase",
    ]
    assert result["dataset"]["attempted_run_count"] == 2
    assert {case["question_id"] for case in result["cases"]} == {
        "d01_seedling_protection", "d02_seedling_paraphrase",
    }


@pytest.mark.parametrize("question_ids", [[], ["missing"], ["g01_llm_definition"] * 2])
def test_runner_rejects_invalid_target_question_ids(tmp_path, question_ids):
    dataset = load_agent_dataset(default_agent_dataset_path())
    environment = environment_for(dataset)
    with pytest.raises(ValueError):
        AgentEvaluationRunner(
            dataset, environment, settings(), "targeted", tmp_path,
            FakeAgentService(dataset, environment),
        ).run(question_ids=question_ids)


def test_variability_and_baseline_comparison_are_deterministic():
    cases = [
        {"question_id": "a", "actual_scope": "general_knowledge", "status": "direct",
         "tool_used": False, "tool_call_count": 0},
        {"question_id": "a", "actual_scope": "document_research",
         "status": "insufficient_evidence", "tool_used": True, "tool_call_count": 1},
        {"question_id": "b", "actual_scope": "general_knowledge", "status": "direct",
         "tool_used": False, "tool_call_count": 0},
        {"question_id": "b", "actual_scope": "general_knowledge", "status": "direct",
         "tool_used": False, "tool_call_count": 0},
    ]
    variability = decision_variability(cases)
    assert variability["questions_with_variation"] == 1
    assert variability["stable_question_rate"] == .5
    base = {
        "dataset": {"sha256": "same", "repeats": 2},
        "agent": {"provider": "ollama", "model": "model", "scope_prompt_version": "v1",
                  "document_decision_prompt_version": "d1",
                  "document_decision_schema_version": "s1",
                  "generation_schema_version": "g1", "grounding_prompt_version": "gp1",
                  "grounded_answer_schema_version": "gs1", "temperature": 0,
                  "graph_step_limit": 20, "max_tool_calls": 3, "tool_top_k": 5},
        "retrieval": {"strategy": "semantic", "embedding_profile": "p1",
                      "min_similarity": .15},
        "metrics": {"tool_selection_accuracy": 1, "document_scope_compliance": 1,
                    "error_count": 0, "configured_timeout_overrun_count": 0},
    }
    current = json.loads(json.dumps(base))
    current["metrics"]["tool_selection_accuracy"] = .8
    comparison = compare_agent_results(current, base)
    assert comparison["compatible"] is True and comparison["regression"] is True
    assert comparison["failure_count_deltas"] == {
        "error_count": 0, "configured_timeout_overrun_count": 0,
    }
    current["dataset"]["repeats"] = 3
    assert compare_agent_results(current, base) == {
        "compatible": False, "metric_deltas": {}, "regression": False,
    }


@pytest.mark.parametrize("repeats,strategy", [(0, "semantic"), (21, "semantic"),
                                                (True, "semantic"), (1, "keyword")])
def test_runner_rejects_invalid_repeat_and_strategy_configuration(tmp_path, repeats, strategy):
    dataset = load_agent_dataset(default_agent_dataset_path())
    environment = environment_for(dataset)
    service = FakeAgentService(dataset, environment)
    with pytest.raises(ValueError):
        AgentEvaluationRunner(dataset, environment, settings(), "invalid", tmp_path,
                              service).run(repeats, strategy)
