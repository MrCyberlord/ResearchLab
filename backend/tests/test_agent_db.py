"""Agent document tool against isolated PostgreSQL schemas and real pgvector queries."""
import json
import os
from contextlib import nullcontext
from uuid import uuid4

import pytest

from test_ingestion_db import db
from test_retrieval_db import seed, vector
from app.application.agent import AgentService
from app.application.agent_graph import AgentGraph
from app.application.grounding import GroundedAnswerService
from app.domain.agent import AgentDecision, AgentGeneration
from app.domain.rag import LLMResult, SourceReference
from app.infrastructure.config import Settings
from app.infrastructure.document_search import SessionDocumentSearchTool

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.getenv("RUN_DB_TESTS") != "1",
    reason="Set RUN_DB_TESTS=1 to use migrated local PostgreSQL",
)]


class Decisions:
    provider = "test-agent"
    model = "test-model"

    def __init__(self, values):
        self.values = list(values)

    def decide(self, question, observations, calls_remaining, *, timeout_seconds=None):
        return self.values.pop(0)

    def generate(self, system_prompt, user_prompt, *, timeout_seconds=None):
        return AgentGeneration("general")


class EvidenceLLM:
    provider = "test-grounding"
    model = "test-model"

    def generate(self, system_prompt, user_prompt, source_ids, *, timeout_seconds=None):
        source = json.loads(user_prompt)["sources"][0]
        return LLMResult(
            "The selected manual states the answer [S1].",
            [SourceReference(source["source_id"], source["text"])], False,
        )


def agent(db, decisions):
    settings = Settings(
        _env_file=None, embedding_provider="local", embedding_model="local-hash-v1",
        retrieval_min_similarity=-1,
    )
    graph = AgentGraph(
        Decisions(decisions), SessionDocumentSearchTool(lambda: nullcontext(db), settings),
        GroundedAnswerService(EvidenceLLM()),
    )
    return AgentService(graph)


def test_agent_tool_preserves_postgresql_document_filter_and_citation_mapping(db):
    seed(db, "excluded.pdf", [(1, "Excluded document evidence.", vector())])
    selected = seed(db, "selected.pdf", [(
        4, "The selected calibration interval is ninety days.", vector(.8, .6),
    )])
    result = agent(db, [
        AgentDecision("search_documents", "calibration interval", 3),
        AgentDecision("answer_grounded"),
    ]).run("What interval does the selected manual specify?", [selected.id], "semantic")
    assert result.status == "grounded" and len(result.citations) == 1
    citation = result.citations[0]
    assert citation.document_id == selected.id
    assert citation.filename == "selected.pdf" and citation.page_number == 4
    assert citation.quote == "The selected calibration interval is ninety days."
    assert result.metadata.tool_activity[0].result_count == 1


def test_agent_tool_missing_document_id_returns_insufficient_without_fabrication(db):
    seed(db, "existing.pdf", [(1, "Existing evidence.", vector())])
    result = agent(db, [
        AgentDecision("search_documents", "missing evidence", 3),
        AgentDecision("insufficient_evidence"),
    ]).run("What does the missing manual say?", [uuid4()], "hybrid")
    assert result.status == "insufficient_evidence"
    assert result.answer is None and result.citations == []
    assert result.metadata.tool_activity[0].outcome == "empty"
