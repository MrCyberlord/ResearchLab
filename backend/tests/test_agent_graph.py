"""Deterministic tests execute the real LangGraph with an in-memory provider."""
import pytest

from app.application.agent import AgentService
from app.application.agent_graph import AgentGraph, choose_route
from app.domain.agent import (
    AgentExecutionLimit, AgentGeneration, AgentLLMInvalidResponse, AgentLLMUnavailable,
    AgentQuestionInvalid, AgentTokenUsage,
)


class RecordingAgentProvider:
    provider = "test-provider"
    model = "test-model"

    def __init__(self, failure=None):
        self.failure = failure
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.failure:
            raise self.failure
        return AgentGeneration("A real provider response.", AgentTokenUsage(12, 5))


def service(provider=None, step_limit=8):
    provider = provider or RecordingAgentProvider()
    return AgentService(AgentGraph(provider), graph_step_limit=step_limit), provider


@pytest.mark.parametrize("question,route", [
    ("Explain cosine similarity.", "direct"),
    ("Summarize my uploaded PDF.", "document_tools_unavailable"),
    ("What does the selected document conclude?", "document_tools_unavailable"),
])
def test_routing_is_deterministic(question, route):
    assert choose_route(question) == route
    assert choose_route(question) == route


def test_direct_graph_transition_and_single_generation_boundary():
    application, provider = service()
    result = application.run("Explain cosine similarity.")
    assert result.answer == "A real provider response."
    assert result.metadata.route == "direct"
    assert [item.node for item in result.metadata.node_trace] == ["route", "direct_response", "finalize"]
    assert result.metadata.steps == 3
    assert result.metadata.usage == AgentTokenUsage(12, 5)
    assert len(provider.calls) == 1
    assert "no tools" in provider.calls[0][0].lower()
    assert provider.calls[0][1] == "Explain cosine similarity."
    assert result.document_tools_available is False


def test_document_route_states_tool_boundary_without_fabricating_grounding():
    application, provider = service()
    result = application.run("Summarize my uploaded PDF.")
    assert result.metadata.route == "document_tools_unavailable"
    assert [item.node for item in result.metadata.node_trace] == [
        "route", "document_tools_unavailable_response", "finalize",
    ]
    assert len(provider.calls) == 1
    system_prompt = provider.calls[0][0].lower()
    assert "document-search tools are not available" in system_prompt
    assert "do not claim to inspect" in system_prompt


def test_metadata_can_be_omitted_without_changing_execution():
    application, provider = service()
    result = application.run("What is an embedding?", include_metadata=False)
    assert result.metadata is None
    assert result.answer == "A real provider response."
    assert len(provider.calls) == 1


@pytest.mark.parametrize("question", ["", " \t\n ", "\x00", "x" * 2001])
def test_invalid_question_never_crosses_provider_boundary(question):
    application, provider = service()
    with pytest.raises(AgentQuestionInvalid):
        application.run(question)
    assert provider.calls == []


def test_provider_failure_propagates_without_fallback_answer():
    failure = AgentLLMUnavailable("provider offline")
    application, provider = service(RecordingAgentProvider(failure))
    with pytest.raises(AgentLLMUnavailable):
        application.run("Explain vectors.")
    assert len(provider.calls) == 1


def test_provider_contract_rejects_blank_answer_without_fallback():
    provider = RecordingAgentProvider()
    provider.generate = lambda system_prompt, user_prompt: AgentGeneration("   ")
    application = AgentService(AgentGraph(provider))
    with pytest.raises(AgentLLMInvalidResponse):
        application.run("Explain vectors.")


def test_graph_step_limit_is_enforced():
    application, provider = service(step_limit=2)
    with pytest.raises(AgentExecutionLimit):
        application.run("Explain vectors.")
    assert len(provider.calls) <= 1


def test_agent_provider_does_not_require_rag_source_ids():
    application, provider = service()
    application.run("Give a short definition of retrieval.")
    assert len(provider.calls[0]) == 2
