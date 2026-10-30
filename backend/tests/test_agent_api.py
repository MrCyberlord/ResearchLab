from fastapi.testclient import TestClient
import pytest

from app.api.dependencies import get_agent_service
from app.application.agent import AgentService
from app.application.agent_graph import AgentGraph
from app.domain.agent import AgentGeneration, AgentLLMInvalidResponse, AgentLLMTimeout, AgentLLMUnavailable
from app.infrastructure.config import Settings
from app.main import create_app


class APIProvider:
    provider = "test-provider"
    model = "test-model"
    def __init__(self, failure=None): self.failure = failure
    def generate(self, system_prompt, user_prompt):
        if self.failure: raise self.failure
        return AgentGeneration("An actual test-provider response.")


def client(provider=None):
    app = create_app(Settings(_env_file=None, database_url=None))
    app.dependency_overrides[get_agent_service] = lambda: AgentService(AgentGraph(provider or APIProvider()))
    return TestClient(app)


def test_agent_config_exposes_limits_and_no_tools():
    with client() as http:
        response = http.get("/api/agent/config")
    assert response.status_code == 200
    assert response.json()["document_tools_available"] is False
    assert response.json()["provider"] == "ollama"
    assert response.json()["graph_step_limit"] == 8


def test_agent_question_response_is_explicitly_ungrounded_with_trace():
    with client() as http:
        response = http.post("/api/agent/questions", json={"question": "Explain embeddings."})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "An actual test-provider response."
    assert body["document_tools_available"] is False
    assert "not document-grounded" in body["disclaimer"]
    assert body["metadata"]["route"] == "direct"
    assert [item["node"] for item in body["metadata"]["node_trace"]] == ["route", "direct_response", "finalize"]


def test_agent_metadata_is_optional():
    with client() as http:
        response = http.post("/api/agent/questions", json={"question": "Explain embeddings.", "include_metadata": False})
    assert response.status_code == 200 and response.json()["metadata"] is None


@pytest.mark.parametrize("body", [
    {}, {"question": ""}, {"question": 12}, {"question": "valid", "unknown": True},
    {"question": "valid", "include_metadata": "yes"},
])
def test_api_validation_rejects_invalid_requests(body):
    with client() as http:
        assert http.post("/api/agent/questions", json=body).status_code == 422


@pytest.mark.parametrize("failure,status,code", [
    (AgentLLMUnavailable(), 503, "agent_llm_unavailable"),
    (AgentLLMTimeout(), 504, "agent_llm_timeout"),
    (AgentLLMInvalidResponse(), 502, "invalid_agent_response"),
])
def test_provider_errors_are_structured(failure, status, code):
    with client(APIProvider(failure)) as http:
        response = http.post("/api/agent/questions", json={"question": "Explain embeddings."})
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code


def test_agent_json_body_limit_is_enforced():
    with client() as http:
        response = http.post("/api/agent/questions", content=b"{" + b" " * 33000,
                             headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "request_too_large"
