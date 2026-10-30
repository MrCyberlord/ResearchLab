"""Lazy production document-search adapter; direct agent paths do not require PostgreSQL."""
from uuid import UUID

from app.application.retrieval import RetrievalService
from app.domain.agent import AgentToolInvalid
from app.domain.rag import QuestionInvalid, RetrievalResult, RetrievalStrategy, RetrievalUnavailable
from app.infrastructure.config import Settings
from app.infrastructure.embeddings import create_embedding_provider
from app.infrastructure.reranker import create_reranker_provider
from app.infrastructure.retrieval_repository import SQLAlchemyRetrievalRepository


def build_retrieval_service(settings: Settings, repository) -> RetrievalService:
    return RetrievalService(
        repository, create_embedding_provider(settings),
        settings.retrieval_top_k, settings.retrieval_min_similarity,
        default_strategy=settings.retrieval_strategy,
        semantic_candidates=settings.semantic_candidate_count,
        keyword_candidates=settings.keyword_candidate_count,
        rrf_constant=settings.rrf_fusion_constant,
        reranker=create_reranker_provider(settings),
        reranker_candidates=settings.reranker_candidate_count,
    )


class SessionDocumentSearchTool:
    """Open a short-lived session only when LangGraph selects document search."""
    name = "document_search"

    def __init__(self, session_factory, settings: Settings):
        self.session_factory = session_factory
        self.settings = settings

    def search(self, query: str, top_k: int, document_ids: list[UUID] | None,
               strategy: RetrievalStrategy | None) -> RetrievalResult:
        if self.session_factory is None:
            raise RetrievalUnavailable("Database is not configured for document search.")
        with self.session_factory() as session:
            service = build_retrieval_service(
                self.settings, SQLAlchemyRetrievalRepository(session),
            )
            try:
                return service.retrieve(query, top_k, document_ids, strategy)
            except QuestionInvalid as exc:
                raise AgentToolInvalid(
                    "The model produced invalid document-search arguments.",
                ) from exc
