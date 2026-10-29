"""Retrieval/answer contracts; no framework or provider imports."""
from dataclasses import dataclass
from typing import Literal
from uuid import UUID


RetrievalStrategy = Literal["semantic", "hybrid", "hybrid_rerank"]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: UUID
    filename: str
    page_number: int
    chunk_order: int
    text: str
    distance: float

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    embedding_profile: str
    retrieval_ms: float
    strategy: RetrievalStrategy = "semantic"


@dataclass(frozen=True)
class SourceReference:
    source_id: str
    quote: str


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class LLMResult:
    answer: str
    evidence: list[SourceReference]
    insufficient_evidence: bool
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class Citation:
    source_id: str
    chunk_id: int
    document_id: UUID
    filename: str
    page_number: int
    chunk_order: int
    quote: str
    text: str


@dataclass(frozen=True)
class RagResult:
    status: Literal["answered", "insufficient_evidence"]
    answer: str | None
    message: str | None
    citations: list[Citation]
    retrieval_ms: float
    generation_ms: float
    retrieved_chunk_count: int
    context_chunk_count: int
    embedding_profile: str
    llm_provider: str
    llm_model: str
    usage: TokenUsage | None
    retrieval_strategy: RetrievalStrategy = "semantic"


class QuestionInvalid(Exception):
    pass


class RetrievalUnavailable(Exception):
    pass


class RerankerUnavailable(Exception):
    pass


class LLMUnavailable(Exception):
    pass


class LLMTimeout(Exception):
    pass


class LLMInvalidResponse(Exception):
    pass
