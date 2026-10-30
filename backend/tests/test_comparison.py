"""Deterministic comparison-engine and API contracts; no external inference."""
from dataclasses import asdict
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from app.api.dependencies import get_comparison_service
from app.application.comparison import ComparisonService
from app.domain.agent import (
    AgentExecutionMetadata, AgentLLMInvalidResponse, AgentNodeTrace, AgentResult,
    AgentTokenUsage, AgentToolActivity,
)
from app.domain.rag import Citation, LLMTimeout, RagResult, TokenUsage
from app.infrastructure.config import Settings
from app.main import create_app


def citation():
    return Citation(
        "S1", 42, uuid4(), "manual.pdf", 3, 7,
        "The warranty lasts three years.",
        "PRIVATE FULL CHUNK: The warranty lasts three years.",
    )


def rag_result(status="answered"):
    answered = status == "answered"
    return RagResult(
        status, "Three years [S1]." if answered else None,
        None if answered else "No evidence.", [citation()] if answered else [],
        4.5, 8.5, 2, 1, "profile", "ollama", "gemma3:4b",
        TokenUsage(20, 5), "hybrid",
    )


def agent_result(status="grounded"):
    grounded = status == "grounded"
    direct = status == "direct"
    activity = [] if direct else [AgentToolActivity(
        1, "PRIVATE raw model query", 3, 2, 1.5, "results",
        "0123456789abcdef", "hybrid", 1, 2, 118000,
    )]
    trace = [AgentNodeTrace(
        "decide", .5, "answer_direct" if direct else "answer_grounded",
        "answer_direct" if direct else "answer_grounded",
        "general_knowledge" if direct else "document_research",
        "general_question" if direct else "evidence_supports_answer",
        0 if direct else 2, 118000,
    )]
    metadata = AgentExecutionMetadata(
        "direct" if direct else "document_research", 9.0, len(trace), len(activity),
        trace, activity, AgentTokenUsage(30, 6), 120000, 118000,
    )
    return AgentResult(
        status,
        "General answer." if direct else "Three years [S1]." if grounded else None,
        "No evidence." if status == "insufficient_evidence" else None,
        [citation()] if grounded else [], "ollama", "gemma3:4b", True, metadata,
    )


class FakeRag:
    def __init__(self, result=None, failure=None, order=None):
        self.result = result or rag_result()
        self.failure = failure
        self.order = order if order is not None else []
        self.calls = []
        self.llm = SimpleNamespace(provider="ollama", model="gemma3:4b")

    def answer(self, question, top_k, document_ids, strategy):
        self.order.append("rag")
        self.calls.append((question, top_k, document_ids, strategy))
        if self.failure:
            raise self.failure
        return self.result


class FakeAgent:
    def __init__(self, result=None, failure=None, order=None):
        self.result = result or agent_result()
        self.failure = failure
        self.order = order if order is not None else []
        self.calls = []
        self.graph = SimpleNamespace(provider=SimpleNamespace(
            provider="ollama", model="gemma3:4b",
        ))
        self.execution_timeout_seconds = 120

    def run(self, question, document_ids, strategy, include_metadata):
        self.order.append("agentic")
        self.calls.append((question, document_ids, strategy, include_metadata))
        if self.failure:
            raise self.failure
        return self.result


def test_comparison_runs_sequentially_and_forwards_identical_scope_and_strategy():
    order = []
    rag = FakeRag(order=order)
    agent = FakeAgent(order=order)
    document_ids = [uuid4()]
    result = ComparisonService(rag, agent).compare(
        "How long is the warranty?", document_ids, "hybrid",
    )
    assert order == ["rag", "agentic"]
    assert rag.calls == [("How long is the warranty?", None, document_ids, "hybrid")]
    assert agent.calls == [("How long is the warranty?", document_ids, "hybrid", True)]
    assert result.execution_mode == "sequential"
    assert result.document_scope == "selected_documents"
    assert result.rag.status == "answered" and result.rag.grounding_mode == "document_grounded"
    assert result.agentic.status == "grounded"
    assert result.agentic.grounding_mode == "document_grounded"
    assert result.rag.citations[0].quote == "The warranty lasts three years."
    assert not hasattr(result.rag.citations[0], "text")
    assert result.agentic.tool_call_count == 1
    assert not hasattr(result.agentic.tool_activity[0], "query")
    assert result.agentic.tool_activity[0].query_fingerprint == "0123456789abcdef"


def test_general_agent_answer_is_distinct_from_document_grounded_rag():
    result = ComparisonService(
        FakeRag(), FakeAgent(agent_result("direct")),
    ).compare("What is an LLM?", None, "semantic")
    assert result.document_scope == "all_documents"
    assert result.rag.grounding_mode == "document_grounded"
    assert result.agentic.status == "direct"
    assert result.agentic.grounding_mode == "general_knowledge"
    assert result.agentic.route == "direct"
    assert result.agentic.citations == [] and result.agentic.tool_call_count == 0


def test_empty_document_selection_is_forwarded_without_broadening():
    rag = FakeRag(rag_result("insufficient_evidence"))
    agent = FakeAgent(agent_result("insufficient_evidence"))
    result = ComparisonService(rag, agent).compare("Question?", [], None)
    assert result.document_scope == "no_documents"
    assert rag.calls[0][2] == [] and agent.calls[0][1] == []
    assert result.rag.status == result.agentic.status == "insufficient_evidence"
    assert result.agentic.grounding_mode == "insufficient_document_evidence"


def test_rag_failure_does_not_prevent_agent_success_and_error_is_sanitized():
    order = []
    result = ComparisonService(
        FakeRag(failure=LLMTimeout("PRIVATE upstream diagnostic"), order=order),
        FakeAgent(agent_result("direct"), order=order),
    ).compare("Explain vectors.")
    assert order == ["rag", "agentic"]
    assert result.rag.status == "error"
    assert result.rag.error.code == "pipeline_timeout" and result.rag.error.retryable
    assert "PRIVATE" not in result.rag.error.message
    assert result.agentic.status == "direct"


def test_agent_failure_preserves_only_safe_partial_metadata_after_rag_success():
    failure = AgentLLMInvalidResponse("PRIVATE response body")
    failure.agent_trace = [AgentNodeTrace(
        "decide", 1.0, "failed", error_type="AgentLLMInvalidResponse",
        error_code="provider_schema", remaining_budget_ms=5000,
    )]
    failure.agent_tool_activity = [AgentToolActivity(
        1, "PRIVATE query", 2, 0, .5, "empty", "fedcba9876543210",
        "semantic", None, 0, 6000,
    )]
    failure.agent_route = "document_research"
    failure.agent_tool_call_count = 1
    failure.agent_remaining_budget_ms = 5000
    result = ComparisonService(FakeRag(), FakeAgent(failure=failure)).compare("Question?")
    assert result.rag.status == "answered"
    assert result.agentic.status == "error"
    assert result.agentic.error.code == "invalid_model_response"
    assert result.agentic.route == "document_research"
    assert result.agentic.graph_steps == result.agentic.tool_call_count == 1
    assert result.agentic.tool_activity[0].query_fingerprint == "fedcba9876543210"
    serialized = str(asdict(result.agentic))
    assert "PRIVATE" not in serialized


@pytest.fixture
def comparison_client():
    service = ComparisonService(FakeRag(), FakeAgent())
    app = create_app(Settings(_env_file=None, database_url=None))
    app.dependency_overrides[get_comparison_service] = lambda: service
    with TestClient(app) as client:
        yield client, service


def test_comparison_api_serializes_both_results_without_full_chunks_or_raw_queries(
        comparison_client):
    client, _ = comparison_client
    document_id = uuid4()
    response = client.post("/api/compare/questions", json={
        "question": "How long is the warranty?",
        "document_ids": [str(document_id)], "strategy": "hybrid",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["execution_mode"] == "sequential"
    assert body["activity_metrics_are_quality_measures"] is False
    assert body["rag"]["status"] == "answered"
    assert body["agentic"]["status"] == "grounded"
    assert "text" not in body["rag"]["citations"][0]
    assert "query" not in body["agentic"]["tool_activity"][0]
    assert body["agentic"]["node_trace"][0]["decision_reason"] == "evidence_supports_answer"


def test_comparison_api_returns_200_with_independent_partial_failure():
    service = ComparisonService(
        FakeRag(failure=LLMTimeout("PRIVATE upstream diagnostic")),
        FakeAgent(agent_result("direct")),
    )
    app = create_app(Settings(_env_file=None, database_url=None))
    app.dependency_overrides[get_comparison_service] = lambda: service
    with TestClient(app) as client:
        response = client.post("/api/compare/questions", json={
            "question": "What is an embedding?", "strategy": "semantic",
        })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rag"]["status"] == "error"
    assert body["rag"]["error"] == {
        "code": "pipeline_timeout", "category": "timeout",
        "message": "The bounded pipeline timed out.", "retryable": True,
    }
    assert body["agentic"]["status"] == "direct"
    assert "PRIVATE" not in response.text


@pytest.mark.parametrize("payload", [
    {}, {"question": ""}, {"question": 12}, {"question": "valid", "extra": True},
    {"question": "valid", "strategy": "keyword"},
    {"question": "valid", "document_ids": ["not-a-uuid"]},
    {"question": "bad\u0000question"}, {"question": "x" * 2001},
])
def test_comparison_api_validation_is_strict(comparison_client, payload):
    client, service = comparison_client
    before = (len(service.rag.calls), len(service.agent.calls))
    assert client.post("/api/compare/questions", json=payload).status_code == 422
    assert (len(service.rag.calls), len(service.agent.calls)) == before


def test_comparison_json_body_limit_is_enforced(comparison_client):
    client, _ = comparison_client
    response = client.post(
        "/api/compare/questions", content=b"{" + b" " * 33000,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "request_too_large"
