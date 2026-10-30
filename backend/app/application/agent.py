"""Application service for bounded tool-using Agentic execution."""
import logging
from time import perf_counter
from uuid import UUID

from langgraph.errors import GraphRecursionError

from app.application.agent_graph import AgentGraph
from app.domain.agent import (
    AgentExecutionLimit, AgentExecutionMetadata, AgentQuestionInvalid,
    AgentResult, AgentTokenUsage,
)
from app.domain.rag import RetrievalStrategy

STRATEGIES = {"semantic", "hybrid", "hybrid_rerank"}


def _aggregate_usage(usages: list[AgentTokenUsage]) -> AgentTokenUsage | None:
    if not usages:
        return None

    def total(field: str) -> int | None:
        values = [getattr(usage, field) for usage in usages]
        return sum(values) if all(value is not None for value in values) else None

    prompt = total("prompt_tokens")
    completion = total("completion_tokens")
    return AgentTokenUsage(prompt, completion) if prompt is not None or completion is not None else None


class AgentService:
    def __init__(self, graph: AgentGraph, max_question_chars: int = 2000,
                 graph_step_limit: int = 24, execution_timeout_seconds: float = 120):
        self.graph = graph
        self.max_question_chars = max_question_chars
        self.graph_step_limit = graph_step_limit
        self.execution_timeout_seconds = execution_timeout_seconds
        self.logger = logging.getLogger("researchlab")

    def run(self, question: str, document_ids: list[UUID] | None = None,
            strategy: RetrievalStrategy | None = None,
            include_metadata: bool = True) -> AgentResult:
        value = question.strip() if isinstance(question, str) else ""
        if not value:
            raise AgentQuestionInvalid("Question cannot be empty.")
        if len(value) > self.max_question_chars:
            raise AgentQuestionInvalid(f"Question must be at most {self.max_question_chars} characters.")
        if len(value.encode("utf-8")) > self.max_question_chars * 4:
            raise AgentQuestionInvalid("Question exceeds the encoded input limit.")
        if any(ord(character) < 32 and character not in "\n\r\t" for character in value):
            raise AgentQuestionInvalid("Question contains unsupported control characters.")
        if document_ids is not None:
            if len(document_ids) > 100 or any(not isinstance(item, UUID) for item in document_ids):
                raise AgentQuestionInvalid("Select at most 100 valid document IDs.")
            document_ids = list(dict.fromkeys(document_ids))
        if strategy is not None and strategy not in STRATEGIES:
            raise AgentQuestionInvalid("Unknown document retrieval strategy.")

        started = perf_counter()
        try:
            state = self.graph.invoke(
                value, document_ids, strategy, self.graph_step_limit,
                self.execution_timeout_seconds,
            )
        except GraphRecursionError:
            error = AgentExecutionLimit("The agent exceeded its configured graph step limit.")
            self.logger.info("agent_execution_failed", extra={
                "duration_ms": round((perf_counter() - started) * 1000, 3),
                "agent_error_type": type(error).__name__,
                "agent_error_code": "graph_step_limit",
                "outcome": "failed",
            })
            raise error from None
        except Exception as exc:
            partial_trace = getattr(exc, "agent_trace", [])
            partial_tools = getattr(exc, "agent_tool_activity", [])
            self.logger.info("agent_execution_failed", extra={
                "duration_ms": round((perf_counter() - started) * 1000, 3),
                "agent_route": getattr(exc, "agent_route", None),
                "agent_step_count": len(partial_trace),
                "tool_call_count": len(partial_tools),
                "remaining_budget_ms": getattr(exc, "agent_remaining_budget_ms", None),
                "agent_error_type": type(exc).__name__,
                "agent_error_code": getattr(exc, "agent_error_code", None),
                "outcome": "failed",
            })
            raise
        duration_ms = round((perf_counter() - started) * 1000, 3)
        trace = state["trace"]
        tool_activity = state.get("tool_activity", [])
        usage = _aggregate_usage(state.get("usages", []))
        self.logger.info("agent_execution_completed", extra={
            "duration_ms": duration_ms, "agent_route": state["route"],
            "agent_step_count": len(trace), "tool_call_count": len(tool_activity),
            "outcome": state["status"],
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
        })
        metadata = (AgentExecutionMetadata(
            state["route"], duration_ms, len(trace), len(tool_activity),
            trace, tool_activity, usage,
            round(self.execution_timeout_seconds * 1000, 3),
            trace[-1].remaining_budget_ms if trace else None,
        ) if include_metadata else None)
        return AgentResult(
            state["status"], state.get("answer"), state.get("message"),
            state.get("citations", []), self.graph.provider.provider,
            self.graph.provider.model, True, metadata,
        )
