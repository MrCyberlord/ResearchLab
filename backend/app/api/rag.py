from contextlib import contextmanager
from dataclasses import asdict
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.api.dependencies import get_rag_service, get_retrieval_service
from app.application.rag import RagService
from app.application.retrieval import RetrievalService
from app.domain.documents import EmbeddingFailure, IncompatibleEmbeddings
from app.domain.rag import (LLMInvalidResponse, LLMTimeout, LLMUnavailable,
                            QuestionInvalid, RetrievalUnavailable)

router = APIRouter(prefix="/api", tags=["retrieval and RAG"])
QuestionText = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=2000)]


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k: int | None = Field(default=None, ge=1, le=20, strict=True)
    document_ids: list[UUID] | None = Field(default=None, max_length=100)


class RetrievalRequest(SearchFilters):
    query: QuestionText


class QuestionRequest(SearchFilters):
    question: QuestionText


class RetrievedChunkResponse(BaseModel):
    chunk_id: int
    document_id: UUID
    filename: str
    page_number: int
    chunk_order: int
    text: str
    distance: float = Field(ge=0, le=2)
    similarity: float = Field(ge=-1, le=1)


class RetrievalResponse(BaseModel):
    chunks: list[RetrievedChunkResponse]
    embedding_profile: str
    retrieval_ms: float = Field(ge=0)


class CitationResponse(BaseModel):
    source_id: str
    chunk_id: int
    document_id: UUID
    filename: str
    page_number: int
    chunk_order: int
    quote: str
    text: str


class UsageResponse(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


class AnswerResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence"]
    answer: str | None
    message: str | None
    citations: list[CitationResponse]
    retrieval_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    retrieved_chunk_count: int = Field(ge=0)
    context_chunk_count: int = Field(ge=0)
    embedding_profile: str
    llm_provider: str
    llm_model: str
    usage: UsageResponse | None


class RagConfigResponse(BaseModel):
    default_top_k: int
    max_top_k: int = 20
    min_similarity: float
    llm_provider: str
    llm_model: str
    max_question_chars: int = 2000


@contextmanager
def rag_errors():
    try:
        yield
    except QuestionInvalid as exc:
        raise HTTPException(422, detail={"code": "invalid_question", "message": str(exc)}) from exc
    except IncompatibleEmbeddings as exc:
        raise HTTPException(409, detail={"code": "embedding_profile_mismatch", "message": "The configured embedding profile does not match this corpus. Re-embed before querying."}) from exc
    except EmbeddingFailure as exc:
        raise HTTPException(503, detail={"code": "embedding_failed", "message": "Query embedding failed. Check the local MiniLM cache."}) from exc
    except RetrievalUnavailable as exc:
        raise HTTPException(503, detail={"code": "storage_unavailable", "message": "The document store is unavailable. Retry later."}) from exc
    except LLMTimeout as exc:
        raise HTTPException(504, detail={"code": "llm_timeout", "message": "The configured model timed out. Retry later or select a smaller local model."}) from exc
    except LLMUnavailable as exc:
        raise HTTPException(503, detail={"code": "llm_unavailable", "message": "The LLM provider or configured model is unavailable. Check provider setup."}) from exc
    except LLMInvalidResponse as exc:
        raise HTTPException(502, detail={"code": "invalid_llm_response", "message": "The model returned an invalid or unsupported answer. No unverified citations were returned."}) from exc


@router.get("/rag/config", response_model=RagConfigResponse)
def rag_config(request: Request):
    settings = request.app.state.settings
    return RagConfigResponse(default_top_k=settings.retrieval_top_k, min_similarity=settings.retrieval_min_similarity,
                             llm_provider=settings.llm_provider, llm_model=settings.llm_model)


@router.post("/retrieval", response_model=RetrievalResponse)
def retrieve(body: RetrievalRequest, service: RetrievalService = Depends(get_retrieval_service)):
    with rag_errors():
        result = service.retrieve(body.query, body.top_k, body.document_ids)
        return {"chunks": [{**asdict(chunk), "similarity": chunk.similarity} for chunk in result.chunks],
                "embedding_profile": result.embedding_profile, "retrieval_ms": result.retrieval_ms}


@router.post("/rag/questions", response_model=AnswerResponse)
def answer(body: QuestionRequest, service: RagService = Depends(get_rag_service)):
    with rag_errors():
        return asdict(service.answer(body.question, body.top_k, body.document_ids))
