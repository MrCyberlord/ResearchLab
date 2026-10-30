"""Opt-in local Ollama smoke test for the real LangGraph Agentic path."""
import os

import pytest

from app.application.agent import AgentService
from app.application.agent_graph import AgentGraph
from app.infrastructure.agent_llm import create_agent_generation_provider
from app.infrastructure.config import Settings


@pytest.mark.agent_smoke
@pytest.mark.skipif(os.getenv("RUN_AGENT_SMOKE") != "1", reason="Set RUN_AGENT_SMOKE=1 for real local agent inference")
def test_real_agent_graph_with_configured_ollama():
    settings = Settings()
    result = AgentService(AgentGraph(create_agent_generation_provider(settings)),
                          settings.agent_max_question_chars,
                          settings.agent_graph_step_limit).run("In one sentence, what is a vector embedding?")
    assert result.answer.strip()
    assert result.provider == "ollama" and result.model == settings.llm_model
    assert result.metadata.route == "direct" and result.metadata.steps == 3
    assert result.document_tools_available is False
