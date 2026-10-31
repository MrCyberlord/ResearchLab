"""Deterministic evaluation-summary tests; no inference, database or baseline writes."""
from copy import deepcopy
import json

from fastapi.testclient import TestClient
import pytest

from app.api.dependencies import get_evaluation_summary_service
from app.application.evaluation_dashboard import EvaluationSummaryService
from app.domain.evaluation_dashboard import (
    EvaluationArtifactAccessDenied, EvaluationArtifactInvalid,
    EvaluationArtifactMissing,
)
from app.infrastructure.config import Settings
from app.infrastructure.evaluation_artifacts import (
    APPROVED_EVALUATION_ARTIFACTS, VersionedEvaluationArtifactReader,
)
from app.main import create_app


class StubReader:
    def __init__(self, values, missing=()):
        self.values = values
        self.missing = set(missing)

    def read(self, artifact_id):
        if artifact_id in self.missing:
            raise EvaluationArtifactMissing("private path detail")
        return deepcopy(self.values[artifact_id])


@pytest.fixture
def real_reader():
    return VersionedEvaluationArtifactReader()


@pytest.fixture
def artifacts(real_reader):
    return {key: real_reader.read(key) for key in APPROVED_EVALUATION_ARTIFACTS}


def test_real_versioned_artifacts_map_to_complete_safe_summary(real_reader):
    result = EvaluationSummaryService(real_reader).summary()
    assert result["status"] == "complete" and result["issues"] == []
    assert [item["state"] for item in result["artifacts"]] == ["loaded"] * 4
    assert result["rag"]["dataset"]["version"] == "researchlab-rag-eval-v1"
    assert result["rag"]["dataset"]["answerable_question_count"] == 6
    assert [item["strategy"] for item in result["rag"]["strategies"]] == [
        "semantic", "hybrid", "hybrid_rerank",
    ]
    semantic_k3 = result["rag"]["strategies"][0]["metrics"][1]
    assert semantic_k3["k"] == 3 and semantic_k3["recall_at_k"] == 1.0
    assert semantic_k3["answerable_question_count"] == 6
    agent = result["agentic"]
    assert agent["denominators"] == {
        "attempted_runs": 20, "completed_runs": 17, "failed_runs": 3,
        "expected_grounded_runs": 8, "expected_abstention_runs": 6,
        "citation_count": 3, "expected_relevant_reference_count": 8,
        "questions": 10,
    }
    assert agent["failures"]["rate"] == 0.15
    assert agent["evidence_observation"]["retrieved_result_count"] == 2
    assert agent["evidence_observation"]["observed_status"] == "insufficient_evidence"
    assert result["cross_dataset_comparison_allowed"] is False
    forbidden = {"cases", "question_id", "question", "query", "node_trace", "tool_activity", "document_text", "exception"}

    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value)) if value else set()
        return set()

    assert not (keys(result) & forbidden)


def test_reader_rejects_unapproved_identifiers_and_invalid_json(tmp_path):
    reader = VersionedEvaluationArtifactReader(tmp_path)
    with pytest.raises(EvaluationArtifactAccessDenied):
        reader.read("../../.env")
    path = tmp_path / APPROVED_EVALUATION_ARTIFACTS["rag_baseline"]
    path.parent.mkdir(parents=True)
    path.write_text("{not valid JSON", encoding="utf-8")
    with pytest.raises(EvaluationArtifactInvalid):
        reader.read("rag_baseline")


def test_missing_artifact_is_explicit_and_other_section_remains_available(artifacts):
    result = EvaluationSummaryService(StubReader(
        artifacts, missing={"agent_baseline"},
    )).summary()
    assert result["status"] == "partial"
    assert result["rag"] is not None and result["agentic"] is None
    assert {item["code"] for item in result["issues"]} == {"artifact_missing"}
    assert next(item for item in result["artifacts"] if item["artifact_id"] == "agent_baseline")["state"] == "missing"
    assert ".json" not in json.dumps(result) and "private path detail" not in json.dumps(result)


def test_incomplete_primary_and_incompatible_followup_are_reported(artifacts):
    incomplete = deepcopy(artifacts)
    del incomplete["rag_baseline"]["dataset"]["answerable_question_count"]
    result = EvaluationSummaryService(StubReader(incomplete)).summary()
    assert result["status"] == "partial" and result["rag"] is None
    assert any(item["code"] == "artifact_incomplete" for item in result["issues"])

    incompatible = deepcopy(artifacts)
    incompatible["agent_evidence_observation"]["dataset"]["sha256"] = "0" * 64
    result = EvaluationSummaryService(StubReader(incompatible)).summary()
    assert result["status"] == "partial" and result["agentic"] is not None
    assert result["agentic"]["evidence_observation"] is None
    assert any(item["code"] == "artifact_incompatible" for item in result["issues"])


def test_api_serializes_only_aggregate_summary(real_reader):
    service = EvaluationSummaryService(real_reader)
    app = create_app(Settings(_env_file=None, database_url=None))
    app.dependency_overrides[get_evaluation_summary_service] = lambda: service
    with TestClient(app) as client:
        response = client.get("/api/evaluations/summary")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "complete"
    assert body["rag"]["strategies"][2]["strategy"] == "hybrid_rerank"
    assert body["agentic"]["latency"]["sample_count"] == 20
    assert body["generated_from_versioned_artifacts_only"] is True
    assert "cases" not in response.text and "question_id" not in response.text


def test_api_serializes_partial_summary_without_erasing_valid_rag(artifacts):
    service = EvaluationSummaryService(StubReader(
        artifacts, missing={"agent_baseline"},
    ))
    app = create_app(Settings(_env_file=None, database_url=None))
    app.dependency_overrides[get_evaluation_summary_service] = lambda: service
    with TestClient(app) as client:
        response = client.get("/api/evaluations/summary")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "partial"
    assert body["rag"] is not None and body["agentic"] is None
    assert body["issues"] == [{
        "artifact_id": "agent_baseline", "code": "artifact_missing",
        "message": "The approved versioned artifact is missing.",
    }]
