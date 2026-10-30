"""Deterministic tests execute the real compiled tool-using LangGraph."""
import json
from uuid import uuid4

import pytest

from app.application.agent import AgentService
from app.application.agent_graph import AgentGraph
from app.application.grounding import GroundedAnswerService
from app.domain.agent import (
    AgentDecision, AgentExecutionLimit, AgentExecutionTimeout, AgentGeneration,
    AgentLLMInvalidResponse, AgentLLMUnavailable, AgentQuestionInvalid,
    AgentTokenUsage, AgentToolInvalid,
)
from app.domain.rag import (
    LLMInvalidResponse, LLMResult, RetrievalResult, RetrievedChunk, SourceReference, TokenUsage,
)


def chunk(text="The Astral kettle warranty lasts three years.", *, chunk_id=1,
          document_id=None, filename="manual.pdf", page=2, order=0):
    return RetrievedChunk(chunk_id, document_id or uuid4(), filename, page, order, text, .1)


class DecisionProvider:
    provider = "test-provider"
    model = "test-model"

    def __init__(self, decisions, direct_answer="A general answer.", failure=None):
        self.decisions = list(decisions)
        self.direct_answer = direct_answer
        self.failure = failure
        self.decision_calls = []
        self.generation_calls = []

    def decide(self, question, observations, calls_remaining, *, timeout_seconds=None):
        self.decision_calls.append((question, observations, calls_remaining, timeout_seconds))
        if self.failure:
            raise self.failure
        if not self.decisions:
            raise AssertionError("Unexpected extra decision")
        return self.decisions.pop(0)

    def generate(self, system_prompt, user_prompt, *, timeout_seconds=None):
        self.generation_calls.append((system_prompt, user_prompt, timeout_seconds))
        if self.failure:
            raise self.failure
        return AgentGeneration(self.direct_answer, AgentTokenUsage(8, 4))


class RecordingSearchTool:
    def __init__(self, results=(), failure=None):
        self.results = list(results)
        self.failure = failure
        self.calls = []

    def search(self, query, top_k, document_ids, strategy):
        self.calls.append((query, top_k, document_ids, strategy))
        if self.failure:
            raise self.failure
        chunks = self.results.pop(0) if self.results else []
        return RetrievalResult(chunks, "test-profile", 1.2, strategy or "semantic")


class EvidenceLLM:
    provider = "grounding-provider"
    model = "grounding-model"

    def __init__(self, insufficient=False):
        self.insufficient = insufficient
        self.calls = []

    def generate(self, system_prompt, user_prompt, source_ids, *, timeout_seconds=None):
        self.calls.append((system_prompt, user_prompt, source_ids, timeout_seconds))
        if self.insufficient:
            return LLMResult("", [], True, TokenUsage(20, 5))
        source = json.loads(user_prompt)["sources"][0]
        return LLMResult(
            "The warranty lasts three years [S1].",
            [SourceReference(source["source_id"], source["text"])], False,
            TokenUsage(20, 5),
        )


def service(decisions, results=(), *, max_calls=3, step_limit=20,
            failure=None, search_failure=None, grounding_llm=None,
            execution_timeout=120):
    provider = DecisionProvider(decisions, failure=failure)
    tool = RecordingSearchTool(results, search_failure)
    grounding_llm = grounding_llm or EvidenceLLM()
    graph = AgentGraph(
        provider, tool, GroundedAnswerService(grounding_llm),
        max_tool_calls=max_calls, max_tool_top_k=5,
    )
    return (AgentService(graph, graph_step_limit=step_limit,
                         execution_timeout_seconds=execution_timeout),
            provider, tool, grounding_llm)


@pytest.mark.parametrize("question", ["What is an LLM?", "what is llm"])
def test_llm_can_choose_direct_answer_with_zero_tool_calls(question):
    application, provider, tool, grounding = service([
        AgentDecision("answer_direct", usage=AgentTokenUsage(3, 1), scope="general_knowledge"),
    ])
    result = application.run(question)
    assert result.status == "direct" and result.answer == "A general answer."
    assert result.citations == [] and result.message is None
    assert result.metadata.route == "direct" and result.metadata.tool_call_count == 0
    assert [item.node for item in result.metadata.node_trace] == [
        "decide", "direct_response", "finalize",
    ]
    assert result.metadata.usage == AgentTokenUsage(11, 5)
    assert tool.calls == [] and grounding.calls == []
    assert len(provider.decision_calls) == len(provider.generation_calls) == 1


def test_multi_step_search_observation_and_grounded_citation_mapping():
    first = chunk("A related warranty passage.", chunk_id=11, filename="first.pdf")
    support = chunk(chunk_id=22, filename="actual.pdf", page=4)
    document_id = uuid4()
    application, provider, tool, grounding = service([
        AgentDecision("search_documents", "kettle warranty", 2, AgentTokenUsage(4, 2)),
        AgentDecision("search_documents", "Astral warranty duration", 2, AgentTokenUsage(5, 2)),
        AgentDecision("answer_grounded", usage=AgentTokenUsage(6, 2)),
    ], [[first], [support]])
    result = application.run(
        "What is the warranty duration?", [document_id], "hybrid",
    )
    assert result.status == "grounded" and "three" in result.answer
    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.chunk_id == first.chunk_id  # Grounding selected only actual supplied evidence.
    assert citation.filename == "first.pdf" and citation.quote in citation.text
    assert result.metadata.route == "document_research"
    assert result.metadata.tool_call_count == 2
    assert [item.query for item in result.metadata.tool_activity] == [
        "kettle warranty", "Astral warranty duration",
    ]
    assert [item.node for item in result.metadata.node_trace] == [
        "decide", "document_search", "observe",
        "decide", "document_search", "observe",
        "decide", "grounded_response", "finalize",
    ]
    assert all(call[2] == [document_id] and call[3] == "hybrid" for call in tool.calls)
    assert len(provider.decision_calls) == 3 and len(grounding.calls) == 1
    final_observation = provider.decision_calls[-1][1][0]
    assert len(final_observation["searches"]) == 2
    assert len(final_observation["evidence"]) == 2


def test_empty_search_can_end_with_explicit_insufficient_evidence():
    application, provider, tool, grounding = service([
        AgentDecision("search_documents", "missing topic", 3),
        AgentDecision("answer_grounded"),
    ], [[]])
    result = application.run("What do my documents say about the missing topic?")
    assert result.status == "insufficient_evidence"
    assert result.answer is None and result.citations == [] and result.message
    assert result.metadata.tool_call_count == 1
    assert result.metadata.tool_activity[0].outcome == "empty"
    assert grounding.calls == []


@pytest.mark.parametrize("second_query,second_top_k", [
    ("  SAME   query ", 2),
    ("Same query.", 5),
])
def test_repeated_semantically_identical_tool_call_ignores_top_k(
        second_query, second_top_k):
    application, provider, tool, _ = service([
        AgentDecision("search_documents", "same query", 2),
        AgentDecision("search_documents", second_query, second_top_k),
    ], [[chunk()]])
    result = application.run("Research this.")
    assert result.status == "insufficient_evidence"
    assert "repeated" in result.message.lower()
    assert len(tool.calls) == 1 and result.metadata.tool_call_count == 1


def test_tool_call_limit_prevents_unbounded_search_loop():
    application, provider, tool, _ = service([
        AgentDecision("search_documents", "first", 1),
        AgentDecision("search_documents", "second", 1),
    ], [[chunk()]], max_calls=1)
    result = application.run("Research this.")
    assert result.status == "insufficient_evidence"
    assert "limit" in result.message.lower()
    assert len(tool.calls) == 1


def test_evidence_is_deduplicated_by_stable_chunk_id():
    shared = chunk(chunk_id=44)
    application, provider, tool, grounding = service([
        AgentDecision("search_documents", "first", 2),
        AgentDecision("search_documents", "second", 2),
        AgentDecision("answer_grounded"),
    ], [[shared], [shared]])
    result = application.run("Research this.")
    assert result.status == "grounded"
    sources = json.loads(grounding.calls[0][1])["sources"]
    assert len(sources) == 1


def test_grounded_path_rejects_fabricated_citation_labels():
    class FabricatingLLM(EvidenceLLM):
        def generate(self, system_prompt, user_prompt, source_ids, *, timeout_seconds=None):
            return LLMResult(
                "Fabricated claim [S99].",
                [SourceReference("S99", "fabricated quote")], False,
            )
    application, _, _, _ = service([
        AgentDecision("search_documents", "query", 1),
        AgentDecision("answer_grounded"),
    ], [[chunk()]], grounding_llm=FabricatingLLM())
    with pytest.raises(LLMInvalidResponse):
        application.run("Research this.")


@pytest.mark.parametrize("decision", [
    AgentDecision("search_documents", "", 2),
    AgentDecision("search_documents", "query", 0),
    AgentDecision("search_documents", "query", 6),
])
def test_invalid_tool_arguments_fail_before_tool_execution(decision):
    application, _, tool, _ = service([decision])
    with pytest.raises(AgentToolInvalid):
        application.run("Research this.")
    assert tool.calls == []


def test_direct_answer_after_search_is_rejected_as_ungrounded():
    application, _, tool, _ = service([
        AgentDecision("search_documents", "query", 1), AgentDecision("answer_direct"),
    ], [[chunk()]])
    with pytest.raises(AgentLLMInvalidResponse):
        application.run("Research this.")
    assert len(tool.calls) == 1


def test_explicit_document_scope_cannot_take_direct_general_answer_path():
    application, _, tool, _ = service([
        AgentDecision("answer_direct", scope="document_research"),
    ])
    with pytest.raises(AgentLLMInvalidResponse):
        application.run("According to my uploaded PDF, what is an LLM?")
    assert tool.calls == []


def test_general_scope_cannot_invoke_document_tool():
    application, _, tool, _ = service([
        AgentDecision("search_documents", "LLM definition", 2,
                      scope="general_knowledge"),
    ])
    with pytest.raises(AgentLLMInvalidResponse):
        application.run("What is an LLM?")
    assert tool.calls == []


def test_tool_and_provider_failures_propagate_without_fallback_answer():
    application, _, _, _ = service(
        [AgentDecision("answer_direct")], failure=AgentLLMUnavailable("offline"),
    )
    with pytest.raises(AgentLLMUnavailable):
        application.run("General question.")
    application, _, tool, _ = service(
        [AgentDecision("search_documents", "query", 1)],
        search_failure=RuntimeError("storage failed"),
    )
    with pytest.raises(RuntimeError):
        application.run("Research this.")
    assert len(tool.calls) == 1


def test_execution_and_graph_step_limits_are_enforced():
    application, _, _, _ = service(
        [AgentDecision("answer_direct")], execution_timeout=-1,
    )
    with pytest.raises(AgentExecutionTimeout):
        application.run("General question.")
    application, _, _, _ = service(
        [AgentDecision("search_documents", "query", 1)], [[]], step_limit=2,
    )
    with pytest.raises(AgentExecutionLimit):
        application.run("Research this.")


def test_metadata_can_be_omitted_without_disabling_tools():
    application, _, tool, _ = service([
        AgentDecision("search_documents", "query", 1),
        AgentDecision("insufficient_evidence"),
    ], [[]])
    result = application.run("Research this.", include_metadata=False)
    assert result.metadata is None and len(tool.calls) == 1


@pytest.mark.parametrize("question", ["", " \t\n ", "\x00", "x" * 2001])
def test_invalid_question_never_crosses_provider_or_tool_boundary(question):
    application, provider, tool, _ = service([AgentDecision("answer_direct")])
    with pytest.raises(AgentQuestionInvalid):
        application.run(question)
    assert provider.decision_calls == [] and tool.calls == []
