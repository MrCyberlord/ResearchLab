"""Application service for a single bounded Agentic execution."""
import logging
from time import perf_counter

from langgraph.errors import GraphRecursionError

from app.application.agent_graph import AgentGraph
from app.domain.agent import (
    AgentExecutionLimit, AgentExecutionMetadata, AgentQuestionInvalid, AgentResult,
)


class AgentService:
    def __init__(self, graph: AgentGraph, max_question_chars: int = 2000,
                 graph_step_limit: int = 8):
        self.graph = graph
        self.max_question_chars = max_question_chars
        self.graph_step_limit = graph_step_limit
        self.logger = logging.getLogger("researchlab")

    def run(self, question: str, include_metadata: bool = True) -> AgentResult:
        value = question.strip() if isinstance(question, str) else ""
        if not value:
            raise AgentQuestionInvalid("Question cannot be empty.")
        if len(value) > self.max_question_chars:
            raise AgentQuestionInvalid(f"Question must be at most {self.max_question_chars} characters.")
        if len(value.encode("utf-8")) > self.max_question_chars * 4:
            raise AgentQuestionInvalid("Question exceeds the encoded input limit.")
        if any(ord(character) < 32 and character not in "\n\r\t" for character in value):
            raise AgentQuestionInvalid("Question contains unsupported control characters.")

        started = perf_counter()
        try:
            state = self.graph.invoke(value, self.graph_step_limit)
        except GraphRecursionError:
            raise AgentExecutionLimit("The agent exceeded its configured graph step limit.") from None
        duration_ms = round((perf_counter() - started) * 1000, 3)
        route = state["route"]
        trace = state["trace"]
        generation = state["generation"]
        self.logger.info("agent_execution_completed", extra={
            "duration_ms": duration_ms, "agent_route": route,
            "agent_step_count": len(trace), "outcome": "completed",
        })
        metadata = (AgentExecutionMetadata(route, duration_ms, len(trace), trace, generation.usage)
                    if include_metadata else None)
        return AgentResult(state["answer"], self.graph.provider.provider,
                           self.graph.provider.model, False, metadata)
