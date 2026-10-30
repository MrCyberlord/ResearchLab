"""HTTP contract for sequential RAG versus Agentic comparison."""
from dataclasses import asdict
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import (
    BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator,
)

from app.api.agent import AgentNodeTraceResponse
from app.api.dependencies import get_comparison_service
from app.application.comparison import ComparisonService

router = APIRouter(prefix="/api/compare", tags=["comparison"])
QuestionText = Annotated[str, StringConstraints(
    strict=True, strip_whitespace=True, min_length=1, max_length=2000,
)]
Strategy = Literal["semantic", "hybrid", "hybrid_rerank"]
DocumentScope = Literal["all_documents", "selected_documents", "no_documents"]


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: QuestionText
    document_ids: list[UUID] | None = Field(default=None, max_length=100)
    strategy: Strategy | None = None

    @field_validator("question")
    @classmethod
    def safe_question(cls, value: str) -> str:
        if any(ord(character) < 32 and character not in "\n\r\t" for character in value):
            raise ValueError("Question contains unsupported control characters")
        if len(value.encode("utf-8")) > 8000:
            raise ValueError("Question exceeds the encoded input limit")
        return value


class ComparisonErrorResponse(BaseModel):
    code: str = Field(min_length=1, max_length=80)
    category: Literal[
        "validation", "configuration", "embedding", "retrieval", "provider",
        "timeout", "execution", "internal",
    ]
    message: str = Field(min_length=1, max_length=300)
    retryable: bool


class ComparisonCitationResponse(BaseModel):
    source_id: str = Field(pattern=r"^S[1-9]\d*$")
    chunk_id: int = Field(ge=1)
    document_id: UUID
    filename: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    chunk_order: int = Field(ge=0)
    quote: str = Field(min_length=1, max_length=1000)


class ComparisonUsageResponse(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


class SafeToolActivityResponse(BaseModel):
    call_index: int = Field(ge=1)
    top_k: int = Field(ge=1, le=20)
    result_count: int = Field(ge=0)
    duration_ms: float = Field(ge=0)
    outcome: Literal["results", "empty"]
    query_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    retrieval_strategy: Strategy | None = None
    document_filter_count: int | None = Field(default=None, ge=0, le=100)
    evidence_added_count: int | None = Field(default=None, ge=0)
    remaining_budget_ms: float | None = Field(default=None, ge=0)


class RagComparisonResponse(BaseModel):
    pipeline: Literal["rag"]
    status: Literal["answered", "insufficient_evidence", "error"]
    grounding_mode: Literal["document_grounded"]
    document_scope: DocumentScope
    answer: str | None
    message: str | None
    citations: list[ComparisonCitationResponse]
    elapsed_ms: float = Field(ge=0)
    error: ComparisonErrorResponse | None
    provider: str | None
    model: str | None
    retrieval_strategy: Strategy | None
    retrieval_ms: float | None = Field(default=None, ge=0)
    generation_ms: float | None = Field(default=None, ge=0)
    retrieved_chunk_count: int | None = Field(default=None, ge=0)
    context_chunk_count: int | None = Field(default=None, ge=0)
    usage: ComparisonUsageResponse | None

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.status == "error":
            if self.error is None or self.answer is not None or self.citations:
                raise ValueError("RAG errors require only safe error information")
        elif self.error is not None:
            raise ValueError("Successful RAG outcomes cannot include an error")
        elif self.status == "answered" and (not self.answer or not self.citations):
            raise ValueError("Answered RAG outcomes require answer text and citations")
        elif self.status == "insufficient_evidence" and (
            self.answer is not None or not self.message or self.citations
        ):
            raise ValueError("Insufficient RAG outcomes require only a message")
        return self


class AgentComparisonResponse(BaseModel):
    pipeline: Literal["agentic"]
    status: Literal["direct", "grounded", "insufficient_evidence", "error"]
    grounding_mode: Literal[
        "general_knowledge", "document_grounded", "insufficient_document_evidence",
        "unknown",
    ]
    document_scope: DocumentScope
    route: Literal["direct", "document_research"] | None
    answer: str | None
    message: str | None
    citations: list[ComparisonCitationResponse]
    elapsed_ms: float = Field(ge=0)
    error: ComparisonErrorResponse | None
    provider: str | None
    model: str | None
    graph_steps: int | None = Field(default=None, ge=1)
    tool_call_count: int | None = Field(default=None, ge=0)
    node_trace: list[AgentNodeTraceResponse]
    tool_activity: list[SafeToolActivityResponse]
    usage: ComparisonUsageResponse | None
    execution_budget_ms: float | None = Field(default=None, ge=0)
    remaining_budget_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.status == "error":
            if self.error is None or self.answer is not None or self.citations:
                raise ValueError("Agentic errors require only safe error information")
        elif self.error is not None:
            raise ValueError("Successful Agentic outcomes cannot include an error")
        elif self.status == "direct" and (
            not self.answer or self.citations or self.grounding_mode != "general_knowledge"
        ):
            raise ValueError("Direct Agentic outcomes must be ungrounded")
        elif self.status == "grounded" and (
            not self.answer or not self.citations or self.grounding_mode != "document_grounded"
        ):
            raise ValueError("Grounded Agentic outcomes require citations")
        elif self.status == "insufficient_evidence" and (
            self.answer is not None or not self.message or self.citations
            or self.grounding_mode != "insufficient_document_evidence"
        ):
            raise ValueError("Insufficient Agentic outcomes require only a message")
        if self.graph_steps is not None and len(self.node_trace) != self.graph_steps:
            raise ValueError("Agentic trace length must match graph steps")
        if self.tool_call_count is not None and len(self.tool_activity) != self.tool_call_count:
            raise ValueError("Agentic tool activity must match tool-call count")
        return self


class ComparisonResponse(BaseModel):
    execution_mode: Literal["sequential"]
    total_elapsed_ms: float = Field(ge=0)
    document_scope: DocumentScope
    requested_strategy: Strategy | None
    activity_metrics_are_quality_measures: Literal[False] = False
    rag: RagComparisonResponse
    agentic: AgentComparisonResponse


@router.post("/questions", response_model=ComparisonResponse)
def compare(body: ComparisonRequest,
            service: ComparisonService = Depends(get_comparison_service)):
    result = asdict(service.compare(body.question, body.document_ids, body.strategy))
    result["activity_metrics_are_quality_measures"] = False
    return result
