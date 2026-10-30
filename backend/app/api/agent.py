"""HTTP boundary for bounded tool-using Agentic document research."""
from contextlib import contextmanager
from dataclasses import asdict
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.api.dependencies import get_agent_service
from app.application.agent import AgentService
from app.domain.agent import (
    AgentExecutionLimit, AgentExecutionTimeout, AgentLLMInvalidResponse,
    AgentLLMTimeout, AgentLLMUnavailable, AgentQuestionInvalid, AgentToolInvalid,
)
from app.domain.documents import EmbeddingFailure, IncompatibleEmbeddings
from app.domain.rag import (
    LLMInvalidResponse, LLMTimeout, LLMUnavailable, QuestionInvalid,
    RerankerUnavailable, RetrievalUnavailable,
)

router = APIRouter(prefix="/api/agent", tags=["agentic"])
AgentQuestion = Annotated[str, StringConstraints(
    strict=True, strip_whitespace=True, min_length=1, max_length=10000,
)]
Strategy = Literal["semantic", "hybrid", "hybrid_rerank"]


class AgentQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: AgentQuestion
    document_ids: list[UUID] | None = Field(default=None, max_length=100)
    strategy: Strategy | None = None
    include_metadata: bool = Field(default=True, strict=True)


class AgentUsageResponse(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


class AgentNodeTraceResponse(BaseModel):
    node: str
    duration_ms: float = Field(ge=0)
    outcome: str
    decision_action: Literal[
        "answer_direct", "search_documents", "answer_grounded", "insufficient_evidence",
    ] | None = None
    decision_scope: Literal["general_knowledge", "document_research"] | None = None
    decision_reason: Literal[
        "general_question", "evidence_needed", "refine_search",
        "evidence_supports_answer", "evidence_does_not_support_answer",
        "search_budget_exhausted",
    ] | None = None
    evidence_count: int | None = Field(default=None, ge=0)
    remaining_budget_ms: float | None = Field(default=None, ge=0)
    error_type: str | None = None
    error_code: str | None = None
    validation_retry_count: int | None = Field(default=None, ge=0, le=1)
    validation_failure_code: str | None = None


class AgentToolActivityResponse(BaseModel):
    call_index: int = Field(ge=1)
    query: str
    top_k: int = Field(ge=1, le=20)
    result_count: int = Field(ge=0)
    duration_ms: float = Field(ge=0)
    outcome: Literal["results", "empty"]
    query_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    retrieval_strategy: Strategy | None = None
    document_filter_count: int | None = Field(default=None, ge=0, le=100)
    evidence_added_count: int | None = Field(default=None, ge=0)
    remaining_budget_ms: float | None = Field(default=None, ge=0)


class AgentCitationResponse(BaseModel):
    source_id: str
    chunk_id: int
    document_id: UUID
    filename: str
    page_number: int
    chunk_order: int
    quote: str
    text: str


class AgentMetadataResponse(BaseModel):
    route: Literal["direct", "document_research"]
    duration_ms: float = Field(ge=0)
    steps: int = Field(ge=1)
    tool_call_count: int = Field(ge=0)
    node_trace: list[AgentNodeTraceResponse]
    tool_activity: list[AgentToolActivityResponse]
    usage: AgentUsageResponse | None
    execution_budget_ms: float | None = Field(default=None, ge=0)
    remaining_budget_ms: float | None = Field(default=None, ge=0)


class AgentQuestionResponse(BaseModel):
    status: Literal["direct", "grounded", "insufficient_evidence"]
    answer: str | None
    message: str | None
    citations: list[AgentCitationResponse]
    provider: str
    model: str
    document_tools_available: Literal[True]
    disclaimer: str
    metadata: AgentMetadataResponse | None

    @model_validator(mode="after")
    def validate_grounding_shape(self):
        if self.status == "direct" and (not self.answer or self.citations):
            raise ValueError("Direct answers require text and cannot contain citations")
        if self.status == "grounded" and (not self.answer or not self.citations):
            raise ValueError("Grounded answers require text and citations")
        if self.status == "insufficient_evidence" and (
            self.answer is not None or not self.message or self.citations
        ):
            raise ValueError("Insufficient-evidence results require only a message")
        return self


class AgentConfigResponse(BaseModel):
    provider: str
    model: str
    max_question_chars: int
    graph_step_limit: int
    execution_timeout_seconds: float
    max_tool_calls: int
    tool_top_k: int
    max_evidence_chunks: int
    default_strategy: Strategy
    available_strategies: list[Strategy]
    document_tools_available: Literal[True] = True


@contextmanager
def agent_errors():
    try:
        yield
    except (AgentQuestionInvalid, QuestionInvalid) as exc:
        raise HTTPException(422, detail={"code": "invalid_question", "message": str(exc)}) from exc
    except AgentToolInvalid as exc:
        raise HTTPException(502, detail={"code": "invalid_tool_call", "message": str(exc)}) from exc
    except IncompatibleEmbeddings as exc:
        raise HTTPException(409, detail={
            "code": "embedding_profile_mismatch",
            "message": "The configured embedding profile does not match this corpus.",
        }) from exc
    except EmbeddingFailure as exc:
        raise HTTPException(503, detail={
            "code": "embedding_failed", "message": "Document-search embedding failed.",
        }) from exc
    except RetrievalUnavailable as exc:
        raise HTTPException(503, detail={
            "code": "storage_unavailable", "message": "The document store is unavailable.",
        }) from exc
    except RerankerUnavailable as exc:
        raise HTTPException(503, detail={
            "code": "reranker_unavailable", "message": "The configured local reranker is unavailable.",
        }) from exc
    except (AgentExecutionTimeout, AgentLLMTimeout, LLMTimeout) as exc:
        raise HTTPException(504, detail={
            "code": "agent_timeout", "message": "The bounded Agentic execution timed out.",
        }) from exc
    except (AgentLLMUnavailable, LLMUnavailable) as exc:
        raise HTTPException(503, detail={
            "code": "agent_llm_unavailable", "message": "The configured local LLM is unavailable.",
        }) from exc
    except (AgentLLMInvalidResponse, LLMInvalidResponse) as exc:
        raise HTTPException(502, detail={
            "code": "invalid_agent_response", "message": "The local model returned an invalid response.",
        }) from exc
    except AgentExecutionLimit as exc:
        raise HTTPException(500, detail={
            "code": "agent_execution_limit", "message": "The bounded graph step limit was reached.",
        }) from exc


@router.get("/config", response_model=AgentConfigResponse)
def agent_config(request: Request):
    settings = request.app.state.settings
    return AgentConfigResponse(
        provider=settings.llm_provider, model=settings.llm_model,
        max_question_chars=settings.agent_max_question_chars,
        graph_step_limit=settings.agent_graph_step_limit,
        execution_timeout_seconds=settings.agent_execution_timeout_seconds,
        max_tool_calls=settings.agent_max_tool_calls,
        tool_top_k=settings.agent_tool_top_k,
        max_evidence_chunks=settings.agent_max_evidence_chunks,
        default_strategy=settings.retrieval_strategy,
        available_strategies=["semantic", "hybrid", "hybrid_rerank"],
    )


@router.post("/questions", response_model=AgentQuestionResponse)
def ask_agent(body: AgentQuestionRequest, service: AgentService = Depends(get_agent_service)):
    with agent_errors():
        result = service.run(
            body.question, body.document_ids, body.strategy, body.include_metadata,
        )
        disclaimer = {
            "direct": "General model answer; no document evidence or citations were used.",
            "grounded": "Document-grounded answer; review the validated supporting citations.",
            "insufficient_evidence": "No document-grounded answer was produced.",
        }[result.status]
        return {**asdict(result), "disclaimer": disclaimer}
