"""Framework-free result contracts for sequential RAG/Agentic comparison."""
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from app.domain.agent import AgentNodeTrace
from app.domain.rag import RetrievalStrategy

DocumentScope = Literal["all_documents", "selected_documents", "no_documents"]


@dataclass(frozen=True)
class ComparisonCitation:
    source_id: str
    chunk_id: int
    document_id: UUID
    filename: str
    page_number: int
    chunk_order: int
    quote: str


@dataclass(frozen=True)
class ComparisonError:
    code: str
    category: Literal[
        "validation", "configuration", "embedding", "retrieval", "provider",
        "timeout", "execution", "internal",
    ]
    message: str
    retryable: bool


@dataclass(frozen=True)
class ComparisonUsage:
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(frozen=True)
class SafeToolActivity:
    call_index: int
    top_k: int
    result_count: int
    duration_ms: float
    outcome: Literal["results", "empty"]
    query_fingerprint: str | None
    retrieval_strategy: RetrievalStrategy | None
    document_filter_count: int | None
    evidence_added_count: int | None
    remaining_budget_ms: float | None


@dataclass(frozen=True)
class RagComparisonResult:
    pipeline: Literal["rag"]
    status: Literal["answered", "insufficient_evidence", "error"]
    grounding_mode: Literal["document_grounded"]
    document_scope: DocumentScope
    answer: str | None
    message: str | None
    citations: list[ComparisonCitation]
    elapsed_ms: float
    error: ComparisonError | None
    provider: str | None
    model: str | None
    retrieval_strategy: RetrievalStrategy | None
    retrieval_ms: float | None
    generation_ms: float | None
    retrieved_chunk_count: int | None
    context_chunk_count: int | None
    usage: ComparisonUsage | None


@dataclass(frozen=True)
class AgentComparisonResult:
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
    citations: list[ComparisonCitation]
    elapsed_ms: float
    error: ComparisonError | None
    provider: str | None
    model: str | None
    graph_steps: int | None
    tool_call_count: int | None
    node_trace: list[AgentNodeTrace]
    tool_activity: list[SafeToolActivity]
    usage: ComparisonUsage | None
    execution_budget_ms: float | None
    remaining_budget_ms: float | None


@dataclass(frozen=True)
class ComparisonResult:
    execution_mode: Literal["sequential"]
    total_elapsed_ms: float
    document_scope: DocumentScope
    requested_strategy: RetrievalStrategy | None
    rag: RagComparisonResult
    agentic: AgentComparisonResult
