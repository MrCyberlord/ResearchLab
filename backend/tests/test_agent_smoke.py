"""Opt-in real Ollama smoke tests for direct and tool-using LangGraph paths."""
import os
from uuid import uuid4

import pytest

from app.application.agent import AgentService
from app.application.agent_graph import AgentGraph
from app.application.grounding import GroundedAnswerService
from app.domain.rag import RetrievalResult, RetrievedChunk
from app.infrastructure.agent_llm import create_agent_generation_provider
from app.infrastructure.config import Settings
from app.infrastructure.llm import create_llm_provider


class NoSearchExpected:
    def search(self, *args, **kwargs):
        raise AssertionError("The general question should not call document search")


class SyntheticManualSearch:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k, document_ids, strategy):
        self.calls.append((query, top_k, document_ids, strategy))
        evidence = RetrievedChunk(
            1, uuid4(), "maintenance-manual.pdf", 7, 0,
            "The orbital sensor must be recalibrated every ninety days.", .05,
        )
        return RetrievalResult([evidence], "smoke-profile", 1, strategy or "semantic")


def graph(settings, tool):
    return AgentService(AgentGraph(
        create_agent_generation_provider(settings), tool,
        GroundedAnswerService(create_llm_provider(settings),
                              settings.agent_max_context_bytes,
                              settings.agent_max_prompt_bytes),
        max_tool_calls=settings.agent_max_tool_calls,
        max_tool_top_k=settings.agent_tool_top_k,
        max_evidence_chunks=settings.agent_max_evidence_chunks,
        decision_context_bytes=settings.agent_decision_context_bytes,
    ), settings.agent_max_question_chars, settings.agent_graph_step_limit,
       settings.agent_execution_timeout_seconds)


@pytest.mark.agent_smoke
@pytest.mark.skipif(os.getenv("RUN_AGENT_SMOKE") != "1",
                    reason="Set RUN_AGENT_SMOKE=1 for real local agent inference")
@pytest.mark.parametrize("question", ["What is an LLM?", "what is llm"])
def test_real_agent_graph_direct_path_with_configured_ollama(question):
    settings = Settings()
    result = graph(settings, NoSearchExpected()).run(
        question,
    )
    assert result.status == "direct" and result.answer.strip()
    assert result.citations == [] and result.metadata.tool_call_count == 0
    assert result.provider == "ollama" and result.model == settings.llm_model


@pytest.mark.agent_smoke
@pytest.mark.skipif(os.getenv("RUN_AGENT_SMOKE") != "1",
                    reason="Set RUN_AGENT_SMOKE=1 for real local agent inference")
def test_real_agent_graph_document_tool_and_grounded_path():
    settings = Settings()
    tool = SyntheticManualSearch()
    result = graph(settings, tool).run(
        "According to the uploaded maintenance manual, how often must the orbital sensor be recalibrated?",
    )
    assert result.status == "grounded" and result.answer
    assert "ninety" in result.answer.lower() or "90" in result.answer
    assert len(result.citations) >= 1 and result.citations[0].filename == "maintenance-manual.pdf"
    assert result.metadata.tool_call_count >= 1 and tool.calls
