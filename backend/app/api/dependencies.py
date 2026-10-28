from collections.abc import Iterator

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.application.ingestion import IngestionService
from app.infrastructure.document_repository import SQLAlchemyDocumentRepository
from app.infrastructure.embeddings import create_embedding_provider
from app.infrastructure.pdf import IsolatedPDFExtractor
from app.application.retrieval import RetrievalService
from app.application.rag import RagService
from app.infrastructure.retrieval_repository import SQLAlchemyRetrievalRepository
from app.infrastructure.llm import create_llm_provider


def get_session(request: Request) -> Iterator[Session]:
    factory = request.app.state.session_factory
    if factory is None:
        raise HTTPException(status_code=503, detail="Database is not configured")
    with factory() as session:
        yield session


def get_ingestion_service(request: Request, session: Session = Depends(get_session)) -> IngestionService:
    settings = request.app.state.settings
    return IngestionService(
        SQLAlchemyDocumentRepository(session),
        IsolatedPDFExtractor(settings.pdf_max_pages, settings.pdf_max_chars,
                             settings.pdf_timeout_seconds, settings.pdf_memory_mb),
        create_embedding_provider(settings), settings.chunk_size, settings.chunk_overlap, settings.max_chunks,
    )


def get_retrieval_repository(session: Session = Depends(get_session)) -> SQLAlchemyRetrievalRepository:
    return SQLAlchemyRetrievalRepository(session)


def get_retrieval_service(request: Request, repository=Depends(get_retrieval_repository)) -> RetrievalService:
    settings = request.app.state.settings
    return RetrievalService(repository, create_embedding_provider(settings),
                            settings.retrieval_top_k, settings.retrieval_min_similarity)


def get_rag_service(request: Request, retrieval: RetrievalService = Depends(get_retrieval_service)) -> RagService:
    settings = request.app.state.settings
    return RagService(retrieval, create_llm_provider(settings),
                      settings.rag_max_context_bytes, settings.rag_max_prompt_bytes)
