"""Sequential, failure-isolated execution of the existing RAG and Agentic services."""
import logging
from time import perf_counter
from uuid import UUID

from app.application.agent import AgentService
from app.application.rag import RagService
from app.domain.agent import (
    AgentExecutionLimit, AgentExecutionTimeout, AgentLLMInvalidResponse,
    AgentLLMTimeout, AgentLLMUnavailable, AgentQuestionInvalid, AgentToolActivity,
    AgentToolInvalid,
)
from app.domain.comparison import (
    AgentComparisonResult, ComparisonCitation, ComparisonError, ComparisonResult,
    ComparisonUsage, DocumentScope, RagComparisonResult, SafeToolActivity,
)
from app.domain.documents import EmbeddingFailure, IncompatibleEmbeddings
from app.domain.rag import (
    Citation, LLMInvalidResponse, LLMTimeout, LLMUnavailable, QuestionInvalid,
    RerankerUnavailable, RetrievalStrategy, RetrievalUnavailable,
)

logger = logging.getLogger("researchlab")


def _document_scope(document_ids: list[UUID] | None) -> DocumentScope:
    if document_ids is None:
        return "all_documents"
    return "selected_documents" if document_ids else "no_documents"


def _citations(values: list[Citation]) -> list[ComparisonCitation]:
    """Expose validated citation identity/quote, not the full supporting chunk text."""
    return [ComparisonCitation(
        item.source_id, item.chunk_id, item.document_id, item.filename,
        item.page_number, item.chunk_order, item.quote,
    ) for item in values]


def _usage(value) -> ComparisonUsage | None:
    if value is None:
        return None
    return ComparisonUsage(value.prompt_tokens, value.completion_tokens)


def _safe_tool_activity(values: list[AgentToolActivity]) -> list[SafeToolActivity]:
    return [SafeToolActivity(
        item.call_index, item.top_k, item.result_count, item.duration_ms, item.outcome,
        item.query_fingerprint, item.retrieval_strategy, item.document_filter_count,
        item.evidence_added_count, item.remaining_budget_ms,
    ) for item in values]


def _error(exc: Exception) -> ComparisonError:
    """Map known failures without returning exception text or provider diagnostics."""
    if isinstance(exc, (QuestionInvalid, AgentQuestionInvalid)):
        return ComparisonError(
            "invalid_question", "validation", "The shared question is invalid.", False,
        )
    if isinstance(exc, IncompatibleEmbeddings):
        return ComparisonError(
            "embedding_profile_mismatch", "configuration",
            "The configured embedding profile does not match the document corpus.", False,
        )
    if isinstance(exc, EmbeddingFailure):
        return ComparisonError(
            "embedding_failed", "embedding", "Local query embedding failed.", True,
        )
    if isinstance(exc, RetrievalUnavailable):
        return ComparisonError(
            "storage_unavailable", "retrieval", "The document store is unavailable.", True,
        )
    if isinstance(exc, RerankerUnavailable):
        return ComparisonError(
            "reranker_unavailable", "retrieval", "The local reranker is unavailable.", True,
        )
    if isinstance(exc, (AgentExecutionTimeout, AgentLLMTimeout, LLMTimeout)):
        return ComparisonError(
            "pipeline_timeout", "timeout", "The bounded pipeline timed out.", True,
        )
    if isinstance(exc, (AgentLLMUnavailable, LLMUnavailable)):
        return ComparisonError(
            "llm_unavailable", "provider", "The configured local LLM is unavailable.", True,
        )
    if isinstance(exc, (AgentLLMInvalidResponse, LLMInvalidResponse)):
        return ComparisonError(
            "invalid_model_response", "provider",
            "The local model returned an invalid structured response.", False,
        )
    if isinstance(exc, AgentToolInvalid):
        return ComparisonError(
            "invalid_tool_call", "execution", "The model produced an invalid tool call.", False,
        )
    if isinstance(exc, AgentExecutionLimit):
        return ComparisonError(
            "execution_limit", "execution", "The graph step limit was reached.", False,
        )
    return ComparisonError(
        "internal_error", "internal", "The pipeline failed unexpectedly.", False,
    )


class ComparisonService:
    """Run RAG first, then Agentic, while preserving independent outcomes."""

    def __init__(self, rag: RagService, agent: AgentService):
        self.rag = rag
        self.agent = agent

    def _run_rag(self, question: str, document_ids: list[UUID] | None,
                 strategy: RetrievalStrategy | None,
                 scope: DocumentScope) -> RagComparisonResult:
        started = perf_counter()
        try:
            result = self.rag.answer(question, None, document_ids, strategy)
        except Exception as exc:
            elapsed = round((perf_counter() - started) * 1000, 3)
            failure = _error(exc)
            logger.info("comparison_pipeline_completed", extra={
                "comparison_pipeline": "rag", "duration_ms": elapsed,
                "outcome": "error", "comparison_error_code": failure.code,
            })
            llm = getattr(self.rag, "llm", None)
            return RagComparisonResult(
                "rag", "error", "document_grounded", scope,
                None, None, [], elapsed, failure,
                getattr(llm, "provider", None), getattr(llm, "model", None),
                strategy, None, None, None, None, None,
            )
        elapsed = round((perf_counter() - started) * 1000, 3)
        logger.info("comparison_pipeline_completed", extra={
            "comparison_pipeline": "rag", "duration_ms": elapsed,
            "outcome": result.status,
        })
        return RagComparisonResult(
            "rag", result.status, "document_grounded", scope,
            result.answer, result.message, _citations(result.citations), elapsed, None,
            result.llm_provider, result.llm_model, result.retrieval_strategy,
            result.retrieval_ms, result.generation_ms, result.retrieved_chunk_count,
            result.context_chunk_count, _usage(result.usage),
        )

    def _run_agent(self, question: str, document_ids: list[UUID] | None,
                   strategy: RetrievalStrategy | None,
                   scope: DocumentScope) -> AgentComparisonResult:
        started = perf_counter()
        try:
            result = self.agent.run(question, document_ids, strategy, include_metadata=True)
        except Exception as exc:
            elapsed = round((perf_counter() - started) * 1000, 3)
            failure = _error(exc)
            trace = list(getattr(exc, "agent_trace", []))
            activity = list(getattr(exc, "agent_tool_activity", []))
            route = getattr(exc, "agent_route", None)
            graph = getattr(self.agent, "graph", None)
            provider = getattr(graph, "provider", None)
            logger.info("comparison_pipeline_completed", extra={
                "comparison_pipeline": "agentic", "duration_ms": elapsed,
                "outcome": "error", "comparison_error_code": failure.code,
                "agent_step_count": len(trace), "tool_call_count": len(activity),
            })
            return AgentComparisonResult(
                "agentic", "error", "unknown", scope, route,
                None, None, [], elapsed, failure,
                getattr(provider, "provider", None),
                getattr(provider, "model", None),
                len(trace) if trace else None,
                getattr(exc, "agent_tool_call_count", None),
                trace, _safe_tool_activity(activity), None,
                round(getattr(self.agent, "execution_timeout_seconds", 0) * 1000, 3),
                getattr(exc, "agent_remaining_budget_ms", None),
            )
        elapsed = round((perf_counter() - started) * 1000, 3)
        metadata = result.metadata
        grounding_mode = {
            "direct": "general_knowledge",
            "grounded": "document_grounded",
            "insufficient_evidence": "insufficient_document_evidence",
        }[result.status]
        logger.info("comparison_pipeline_completed", extra={
            "comparison_pipeline": "agentic", "duration_ms": elapsed,
            "outcome": result.status,
            "agent_step_count": metadata.steps if metadata else None,
            "tool_call_count": metadata.tool_call_count if metadata else None,
        })
        return AgentComparisonResult(
            "agentic", result.status, grounding_mode, scope,
            metadata.route if metadata else None,
            result.answer, result.message, _citations(result.citations), elapsed, None,
            result.provider, result.model,
            metadata.steps if metadata else None,
            metadata.tool_call_count if metadata else None,
            metadata.node_trace if metadata else [],
            _safe_tool_activity(metadata.tool_activity) if metadata else [],
            _usage(metadata.usage) if metadata else None,
            metadata.execution_budget_ms if metadata else None,
            metadata.remaining_budget_ms if metadata else None,
        )

    def compare(self, question: str, document_ids: list[UUID] | None = None,
                strategy: RetrievalStrategy | None = None) -> ComparisonResult:
        started = perf_counter()
        scope = _document_scope(document_ids)
        # Deliberately sequential: never contend for local Ollama CPU/RAM.
        rag = self._run_rag(question, document_ids, strategy, scope)
        agent = self._run_agent(question, document_ids, strategy, scope)
        return ComparisonResult(
            "sequential", round((perf_counter() - started) * 1000, 3),
            scope, strategy, rag, agent,
        )
