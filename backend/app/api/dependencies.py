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
from app.application.agent import AgentService
from app.application.agent_graph import AgentGraph
from app.application.grounding import GroundedAnswerService
from app.infrastructure.document_search import SessionDocumentSearchTool, build_retrieval_service
from app.infrastructure.agent_llm import create_agent_generation_provider
from app.application.comparison import ComparisonService
from app.application.evaluation_dashboard import EvaluationSummaryService
from app.infrastructure.evaluation_artifacts import VersionedEvaluationArtifactReader


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
    return build_retrieval_service(settings, repository)


def get_rag_service(request: Request, retrieval: RetrievalService = Depends(get_retrieval_service)) -> RagService:
    settings = request.app.state.settings
    return RagService(retrieval, create_llm_provider(settings),
                      settings.rag_max_context_bytes, settings.rag_max_prompt_bytes)


def get_agent_service(request: Request) -> AgentService:
    settings = request.app.state.settings
    graph = AgentGraph(
        create_agent_generation_provider(settings),
        SessionDocumentSearchTool(request.app.state.session_factory, settings),
        GroundedAnswerService(create_llm_provider(settings),
                              settings.agent_max_context_bytes,
                              settings.agent_max_prompt_bytes),
        max_tool_calls=settings.agent_max_tool_calls,
        max_tool_top_k=settings.agent_tool_top_k,
        max_evidence_chunks=settings.agent_max_evidence_chunks,
        decision_context_bytes=settings.agent_decision_context_bytes,
    )
    return AgentService(graph, settings.agent_max_question_chars,
                        settings.agent_graph_step_limit,
                        settings.agent_execution_timeout_seconds)


def get_comparison_service(
    rag: RagService = Depends(get_rag_service),
    agent: AgentService = Depends(get_agent_service),
) -> ComparisonService:
    return ComparisonService(rag, agent)


def get_evaluation_summary_service() -> EvaluationSummaryService:
    return EvaluationSummaryService(VersionedEvaluationArtifactReader())
