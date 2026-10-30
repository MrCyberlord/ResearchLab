"""Agent evaluator isolation against the migrated local PostgreSQL database."""
import os

import pytest
from sqlalchemy import text

from app.evaluation.agent_dataset import default_agent_dataset_path, load_agent_dataset
from app.evaluation.agent_runner import AgentEvaluationRunner
from app.evaluation.runner import IsolatedEvaluationDatabase
from app.infrastructure.config import Settings
from test_agent_evaluation import FakeAgentService

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.getenv("RUN_DB_TESTS") != "1",
    reason="Set RUN_DB_TESTS=1 to use migrated local PostgreSQL",
)]


def test_agent_evaluation_uses_private_rollback_schema_and_preserves_public_corpus(tmp_path):
    dataset = load_agent_dataset(default_agent_dataset_path())
    settings = Settings(
        embedding_provider="local", embedding_model="local-hash-v1",
        retrieval_min_similarity=-1,
    )
    manager = IsolatedEvaluationDatabase(settings, dataset, include_llm=False)
    with manager as environment:
        assert manager.session.scalar(text("SELECT current_schema()")) == environment.schema
        assert environment.schema.startswith("evaluation_")
        assert manager.session.scalar(text("SELECT count(*) FROM documents")) == len(dataset.documents)
        manager.session.commit()
        service = FakeAgentService(dataset, environment)
        result = AgentEvaluationRunner(
            dataset, environment, settings, "isolation", tmp_path, service,
        ).run(repeats=2, strategy="semantic")
        assert result["dataset"]["attempted_run_count"] == len(dataset.questions) * 2
        assert result["metrics"]["tool_selection_accuracy"] == 1
    assert manager.public_counts_after == manager.public_counts_before
    with manager.engine.connect() as connection:
        assert connection.scalar(
            text("SELECT to_regnamespace(:schema)"), {"schema": environment.schema},
        ) is None
