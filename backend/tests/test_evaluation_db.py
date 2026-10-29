import os

import pytest
from sqlalchemy import text

from app.evaluation.dataset import default_dataset_path, load_dataset
from app.evaluation.runner import EvaluationRunner, IsolatedEvaluationDatabase
from app.infrastructure.config import Settings

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.getenv("RUN_DB_TESTS") != "1", reason="Set RUN_DB_TESTS=1 to use migrated local PostgreSQL")]


def test_evaluation_uses_private_rollback_schema_and_preserves_production_counts(tmp_path):
    dataset = load_dataset(default_dataset_path())
    settings = Settings(embedding_provider="local", embedding_model="local-hash-v1")
    manager = IsolatedEvaluationDatabase(settings, dataset, include_llm=False)
    with manager as environment:
        assert manager.session.scalar(text("SELECT current_schema()")) == environment.schema
        assert environment.schema.startswith("evaluation_")
        assert manager.session.scalar(text("SELECT count(*) FROM documents")) == len(dataset.documents)
        manager.session.commit()  # End the inspection transaction before the repository opens its read scope.
        result = EvaluationRunner(dataset, environment, settings, "isolation-test", tmp_path).run([1, 3], 3)
        assert result["dataset"]["version"] == dataset.version
        assert result["retrieval"]["latency_ms"]["sample_count"] == len(dataset.questions)
    assert manager.public_counts_after == manager.public_counts_before
    with manager.engine.connect() as connection:
        assert connection.scalar(text("SELECT to_regnamespace(:schema)"), {"schema": environment.schema}) is None
