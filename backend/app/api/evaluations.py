"""Read-only HTTP contract for aggregate, versioned evaluation evidence."""
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import get_evaluation_summary_service
from app.application.evaluation_dashboard import EvaluationSummaryService

router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


class StrictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactStatusResponse(StrictResponse):
    artifact_id: Literal[
        "rag_baseline", "rag_strategy_reference", "agent_baseline",
        "agent_evidence_observation",
    ]
    state: Literal["loaded", "missing", "invalid", "incomplete", "incompatible"]


class ArtifactIssueResponse(StrictResponse):
    artifact_id: str = Field(min_length=1, max_length=80)
    code: Literal[
        "artifact_missing", "artifact_invalid", "artifact_incomplete",
        "artifact_incompatible",
    ]
    message: str = Field(min_length=1, max_length=240)


class RunResponse(StrictResponse):
    run_id: str = Field(min_length=1, max_length=120)
    created_at_utc: str = Field(min_length=1, max_length=50)


class RagDatasetResponse(StrictResponse):
    version: str
    title: str
    license: str
    document_count: int = Field(ge=1)
    question_count: int = Field(ge=1)
    answerable_question_count: int = Field(ge=1)
    unanswerable_question_count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RagModelResponse(StrictResponse):
    provider: str
    model: str
    revision: str
    dimension: int = Field(ge=1)
    normalization: str


class RagSettingsResponse(StrictResponse):
    default_strategy: Literal["semantic", "hybrid", "hybrid_rerank"]
    k_values: list[int]
    min_similarity: float = Field(ge=0, le=1)
    distance: str
    chunk_size: int = Field(ge=1)
    chunk_overlap: int = Field(ge=0)
    semantic_candidate_count: int | None = Field(default=None, ge=1)
    keyword_candidate_count: int | None = Field(default=None, ge=1)
    rrf_fusion_constant: int | None = Field(default=None, ge=1)


class RagMetricResponse(StrictResponse):
    k: int = Field(ge=1)
    recall_at_k: float = Field(ge=0, le=1)
    precision_at_k: float = Field(ge=0, le=1)
    mrr_at_k: float = Field(ge=0, le=1)
    hit_rate_at_k: float = Field(ge=0, le=1)
    answerable_question_count: int = Field(ge=1)


class LatencyResponse(StrictResponse):
    sample_count: int = Field(ge=1)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)


class RagStrategyResponse(StrictResponse):
    strategy: Literal["semantic", "hybrid", "hybrid_rerank"]
    reference_only: bool
    metrics: list[RagMetricResponse]
    latency: LatencyResponse


class MetricDefinitionResponse(StrictResponse):
    key: str
    label: str
    definition: str
    denominator: str


class RagSummaryResponse(StrictResponse):
    dataset: RagDatasetResponse
    baseline_run: RunResponse
    strategy_reference_run: RunResponse | None
    model: RagModelResponse
    settings: RagSettingsResponse
    strategies: list[RagStrategyResponse]
    metric_definitions: list[MetricDefinitionResponse]
    comparability_note: str


class AgentDatasetResponse(StrictResponse):
    version: str
    title: str
    license: str
    document_count: int = Field(ge=1)
    question_count: int = Field(ge=1)
    repeats: int = Field(ge=1)
    attempted_run_count: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AgentModelResponse(StrictResponse):
    provider: str
    model: str
    temperature: float = Field(ge=0)


class AgentSettingsResponse(StrictResponse):
    scope_prompt_version: str
    decision_prompt_version: str
    decision_schema_version: str
    generation_schema_version: str
    graph_step_limit: int = Field(ge=1)
    max_tool_calls: int = Field(ge=1)
    tool_top_k: int = Field(ge=1)
    execution_timeout_seconds: float = Field(gt=0)
    retrieval_strategy: Literal["semantic", "hybrid", "hybrid_rerank"]
    min_similarity: float = Field(ge=0, le=1)
    embedding_model: str
    embedding_revision: str


class AgentDenominatorsResponse(StrictResponse):
    attempted_runs: int = Field(ge=1)
    completed_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    expected_grounded_runs: int = Field(ge=0)
    expected_abstention_runs: int = Field(ge=0)
    citation_count: int = Field(ge=0)
    expected_relevant_reference_count: int = Field(ge=0)
    questions: int = Field(ge=1)


class AgentMetricResponse(StrictResponse):
    key: str
    label: str
    value: float | None = Field(default=None, ge=0, le=1)
    denominator_name: str
    denominator: int = Field(ge=0)
    definition: str


class DistributionResponse(StrictResponse):
    sample_count: int = Field(ge=0)
    mean: float = Field(ge=0)
    minimum: int = Field(ge=0)
    maximum: int = Field(ge=0)


class AgentActivityResponse(StrictResponse):
    tool_calls: DistributionResponse
    graph_steps: DistributionResponse
    stable_question_rate: float | None = Field(default=None, ge=0, le=1)
    questions_with_variation: int = Field(ge=0)


class FailureResponse(StrictResponse):
    count: int = Field(ge=0)
    rate: float = Field(ge=0, le=1)
    configured_timeout_overrun_count: int = Field(ge=0)


class UsageResponse(StrictResponse):
    samples_with_usage: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class LimitationsResponse(StrictResponse):
    citation_check: str
    factual_entailment_measured: Literal[False]
    sample_size: str


class EvidenceObservationResponse(StrictResponse):
    run: RunResponse
    sample_size: int = Field(ge=1)
    expected_status: str
    observed_status: str
    retrieved_result_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    finding: str
    caution: str


class AgentSummaryResponse(StrictResponse):
    dataset: AgentDatasetResponse
    run: RunResponse
    model: AgentModelResponse
    settings: AgentSettingsResponse
    denominators: AgentDenominatorsResponse
    metrics: list[AgentMetricResponse]
    latency: LatencyResponse
    activity: AgentActivityResponse
    failures: FailureResponse
    usage: UsageResponse
    limitations: LimitationsResponse
    evidence_observation: EvidenceObservationResponse | None
    comparability_note: str


class ResearchFindingResponse(StrictResponse):
    source: Literal[
        "rag_baseline", "agent_baseline", "agent_evidence_observation",
    ]
    title: str
    summary: str
    caution: str


class EvaluationSummaryResponse(StrictResponse):
    schema_version: Literal[1]
    status: Literal["complete", "partial", "unavailable"]
    artifacts: list[ArtifactStatusResponse]
    issues: list[ArtifactIssueResponse]
    rag: RagSummaryResponse | None
    agentic: AgentSummaryResponse | None
    research_findings: list[ResearchFindingResponse]
    cross_dataset_comparison_allowed: Literal[False]
    generated_from_versioned_artifacts_only: Literal[True]


@router.get("/summary", response_model=EvaluationSummaryResponse)
def evaluation_summary(
    service: EvaluationSummaryService = Depends(get_evaluation_summary_service),
):
    return service.summary()
