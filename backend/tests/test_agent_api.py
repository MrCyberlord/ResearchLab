import json
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from app.api.dependencies import get_agent_service
from app.application.agent import AgentService
from app.application.agent_graph import AgentGraph
from app.application.grounding import GroundedAnswerService
from app.domain.agent import (
    AgentDecision, AgentGeneration, AgentLLMInvalidResponse, AgentLLMTimeout,
    AgentLLMUnavailable, AgentToolInvalid,
)
from app.domain.rag import LLMResult, RetrievalResult, RetrievedChunk, SourceReference
from app.domain.documents import EmbeddingFailure, IncompatibleEmbeddings
from app.domain.rag import LLMTimeout, RerankerUnavailable, RetrievalUnavailable
from app.infrastructure.config import Settings
from app.main import create_app


class APIProvider:
    provider = "test-provider"
    model = "test-model"

    def __init__(self, decisions, failure=None):
        self.decisions = list(decisions)
        self.failure = failure

    def decide(self, question, observations, calls_remaining, *, timeout_seconds=None):
        if self.failure:
            raise self.failure
        return self.decisions.pop(0)

    def generate(self, system_prompt, user_prompt, *, timeout_seconds=None):
        if self.failure:
            raise self.failure
        return AgentGeneration("An actual general response.")


class APITool:
    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.calls = []

    def search(self, query, top_k, document_ids, strategy):
        self.calls.append((query, top_k, document_ids, strategy))
        return RetrievalResult(self.chunks, "profile", 1, strategy or "semantic")


class EvidenceLLM:
    provider = "grounding"
    model = "grounding-model"

    def generate(self, system_prompt, user_prompt, source_ids, *, timeout_seconds=None):
        source = json.loads(user_prompt)["sources"][0]
        return LLMResult("Three years [S1].", [SourceReference("S1", source["text"])], False)


def make_service(decisions, chunks=(), failure=None):
    provider = APIProvider(decisions, failure)
    tool = APITool(chunks)
    graph = AgentGraph(provider, tool, GroundedAnswerService(EvidenceLLM()))
    return AgentService(graph), tool


def client(service):
    app = create_app(Settings(_env_file=None, database_url=None))
    app.dependency_overrides[get_agent_service] = lambda: service
    return TestClient(app)


def test_agent_config_exposes_tool_limits_and_strategies():
    service, _ = make_service([AgentDecision("answer_direct")])
    with client(service) as http:
        response = http.get("/api/agent/config")
    assert response.status_code == 200
    body = response.json()
    assert body["document_tools_available"] is True
    assert body["provider"] == "ollama" and body["graph_step_limit"] == 20
    assert body["max_tool_calls"] == 3 and body["tool_top_k"] == 5
    assert body["default_strategy"] == "semantic"
    assert body["available_strategies"] == ["semantic", "hybrid", "hybrid_rerank"]


def test_production_agent_composition_is_lazy_without_database():
    settings = Settings(_env_file=None, database_url=None)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        settings=settings, session_factory=None,
    )))
    service = get_agent_service(request)
    assert service.graph.search_tool.session_factory is None


def test_direct_response_preserves_zero_tool_behavior():
    service, tool = make_service([AgentDecision("answer_direct")])
    with client(service) as http:
        response = http.post("/api/agent/questions", json={"question": "Explain embeddings."})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "direct" and body["answer"] == "An actual general response."
    assert body["citations"] == [] and body["document_tools_available"] is True
    assert "General model answer" in body["disclaimer"]
    assert body["metadata"]["route"] == "direct"
    assert body["metadata"]["tool_call_count"] == 0
    assert tool.calls == []


def test_grounded_response_exposes_citations_and_tool_activity():
    document_id = uuid4()
    evidence = RetrievedChunk(42, document_id, "manual.pdf", 3, 7,
                              "The warranty lasts three years.", .1)
    service, tool = make_service([
        AgentDecision("search_documents", "warranty duration", 3),
        AgentDecision("answer_grounded"),
    ], [evidence])
    with client(service) as http:
        response = http.post("/api/agent/questions", json={
            "question": "How long is the warranty?",
            "document_ids": [str(document_id)], "strategy": "hybrid",
        })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "grounded" and body["answer"] == "Three years [S1]."
    assert body["citations"] == [{
        "source_id": "S1", "chunk_id": 42, "document_id": str(document_id),
        "filename": "manual.pdf", "page_number": 3, "chunk_order": 7,
        "quote": "The warranty lasts three years.",
        "text": "The warranty lasts three years.",
    }]
    assert body["metadata"]["tool_call_count"] == 1
    assert body["metadata"]["tool_activity"][0]["query"] == "warranty duration"
    assert tool.calls[0][2] == [document_id] and tool.calls[0][3] == "hybrid"


def test_missing_document_results_are_explicitly_insufficient():
    service, _ = make_service([
        AgentDecision("search_documents", "missing", 2),
        AgentDecision("insufficient_evidence"),
    ])
    with client(service) as http:
        response = http.post("/api/agent/questions", json={
            "question": "What does the missing document say?",
            "document_ids": [str(uuid4())],
        })
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_evidence"
    assert body["answer"] is None and body["citations"] == [] and body["message"]


def test_agent_metadata_is_optional_without_disabling_execution():
    service, _ = make_service([AgentDecision("answer_direct")])
    with client(service) as http:
        response = http.post("/api/agent/questions", json={
            "question": "Explain embeddings.", "include_metadata": False,
        })
    assert response.status_code == 200 and response.json()["metadata"] is None


@pytest.mark.parametrize("body", [
    {}, {"question": ""}, {"question": 12}, {"question": "valid", "unknown": True},
    {"question": "valid", "include_metadata": "yes"},
    {"question": "valid", "document_ids": ["not-a-uuid"]},
    {"question": "valid", "strategy": "unknown"},
])
def test_api_validation_rejects_invalid_requests(body):
    service, _ = make_service([AgentDecision("answer_direct")])
    with client(service) as http:
        assert http.post("/api/agent/questions", json=body).status_code == 422


class FailingService:
    def __init__(self, failure): self.failure = failure
    def run(self, *args, **kwargs): raise self.failure


@pytest.mark.parametrize("failure,status,code", [
    (AgentLLMUnavailable(), 503, "agent_llm_unavailable"),
    (AgentLLMTimeout(), 504, "agent_timeout"),
    (AgentLLMInvalidResponse(), 502, "invalid_agent_response"),
    (AgentToolInvalid("invalid args"), 502, "invalid_tool_call"),
    (RetrievalUnavailable(), 503, "storage_unavailable"),
    (EmbeddingFailure(), 503, "embedding_failed"),
    (IncompatibleEmbeddings(), 409, "embedding_profile_mismatch"),
    (RerankerUnavailable(), 503, "reranker_unavailable"),
    (LLMTimeout(), 504, "agent_timeout"),
])
def test_agent_errors_are_structured(failure, status, code):
    with client(FailingService(failure)) as http:
        response = http.post("/api/agent/questions", json={"question": "Explain embeddings."})
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code


def test_agent_json_body_limit_is_enforced():
    service, _ = make_service([AgentDecision("answer_direct")])
    with client(service) as http:
        response = http.post("/api/agent/questions", content=b"{" + b" " * 33000,
                             headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "request_too_large"
