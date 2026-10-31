"""Allowlisted, size-bounded reader for committed evaluation JSON artifacts."""
import json
from pathlib import Path

from app.domain.evaluation_dashboard import (
    EvaluationArtifactAccessDenied, EvaluationArtifactInvalid,
    EvaluationArtifactMissing,
)

APPROVED_EVALUATION_ARTIFACTS = {
    "rag_baseline": Path("baselines/minilm-v1.json"),
    "rag_strategy_reference": Path("baselines/hybrid-comparison-v1.json"),
    "agent_baseline": Path("agent/baselines/gemma3-4b-v1.json"),
    "agent_evidence_observation": Path(
        "agent/baselines/gemma3-4b-v1-quality-after-v4.json"
    ),
}


class VersionedEvaluationArtifactReader:
    """Read only fixed repository artifacts; callers cannot provide filesystem paths."""

    def __init__(self, root: Path | None = None, max_bytes: int = 1_000_000):
        self.root = (root or Path(__file__).resolve().parents[2] / "evaluation").resolve()
        self.max_bytes = max_bytes

    def read(self, artifact_id: str) -> dict:
        relative = APPROVED_EVALUATION_ARTIFACTS.get(artifact_id)
        if relative is None:
            raise EvaluationArtifactAccessDenied("Artifact identifier is not approved.")
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise EvaluationArtifactAccessDenied("Artifact escaped the approved root.") from exc
        try:
            size = path.stat().st_size
        except FileNotFoundError as exc:
            raise EvaluationArtifactMissing("Approved artifact is missing.") from exc
        except OSError as exc:
            raise EvaluationArtifactInvalid("Approved artifact is unreadable.") from exc
        if size <= 0 or size > self.max_bytes:
            raise EvaluationArtifactInvalid("Approved artifact size is invalid.")
        try:
            value = json.loads(path.read_text("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvaluationArtifactInvalid("Approved artifact is not valid JSON.") from exc
        if not isinstance(value, dict):
            raise EvaluationArtifactInvalid("Approved artifact root must be an object.")
        return value
