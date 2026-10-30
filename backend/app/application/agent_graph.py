"""Bounded LangGraph orchestration for LLM-driven document research."""
from operator import add
import hashlib
import logging
import re
from time import perf_counter
from typing import Annotated, TypedDict, cast
import unicodedata
from uuid import UUID

from langgraph.graph import END, START, StateGraph

from app.application.grounding import GroundedAnswerService
from app.application.ports import AgentGenerationProvider, DocumentSearchTool
from app.domain.agent import (
    AgentAction, AgentDecision, AgentExecutionTimeout, AgentGeneration,
    AgentLLMInvalidResponse, AgentNodeTrace, AgentTokenUsage, AgentToolActivity,
    AgentToolInvalid,
)
from app.domain.rag import Citation, RetrievalStrategy, RetrievedChunk


class AgentState(TypedDict, total=False):
    question: str
    document_ids: list[UUID] | None
    strategy: RetrievalStrategy | None
    deadline: float
    decision: AgentDecision
    evidence: list[RetrievedChunk]
    attempted_calls: list[str]
    tool_call_count: int
    route: str
    status: str
    answer: str | None
    message: str | None
    citations: list[Citation]
    trace: Annotated[list[AgentNodeTrace], add]
    tool_activity: Annotated[list[AgentToolActivity], add]
    usages: Annotated[list[AgentTokenUsage], add]


DIRECT_SYSTEM_PROMPT = (
    "You are the ResearchLab Agentic workflow. Answer the user's general question clearly and "
    "concisely. This branch has no document evidence. Do not claim to have inspected files, and "
    "do not provide document citations or describe the answer as document-grounded."
)
DEFAULT_INSUFFICIENT = "The selected documents did not provide enough evidence to answer the question."


def _search_identity(query: str) -> str:
    """Canonical query identity; result-count changes do not make a new search."""
    normalized = unicodedata.normalize("NFKC", query).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _query_fingerprint(query: str) -> str:
    """Stable content-free identifier for correlating equivalent search calls."""
    return hashlib.sha256(_search_identity(query).encode("utf-8")).hexdigest()[:16]


class AgentGraph:
    """LLM decision loop with one document-search tool and hard execution bounds."""

    def __init__(self, provider: AgentGenerationProvider, search_tool: DocumentSearchTool,
                 grounding: GroundedAnswerService, *, max_tool_calls: int = 3,
                 max_tool_top_k: int = 5, max_evidence_chunks: int = 12,
                 decision_context_bytes: int = 6000):
        self.provider = provider
        self.search_tool = search_tool
        self.grounding = grounding
        self.max_tool_calls = max_tool_calls
        self.max_tool_top_k = max_tool_top_k
        self.max_evidence_chunks = max_evidence_chunks
        self.decision_context_bytes = decision_context_bytes
        self.logger = logging.getLogger("researchlab")
        builder = StateGraph(AgentState)
        builder.add_node("decide", self._decide)
        builder.add_node("document_search", self._document_search)
        builder.add_node("observe", self._observe)
        builder.add_node("direct_response", self._direct_response)
        builder.add_node("grounded_response", self._grounded_response)
        builder.add_node("insufficient_evidence", self._insufficient_evidence)
        builder.add_node("finalize", self._finalize)
        builder.add_edge(START, "decide")
        builder.add_conditional_edges("decide", self._decision_action, {
            "answer_direct": "direct_response",
            "search_documents": "document_search",
            "answer_grounded": "grounded_response",
            "insufficient_evidence": "insufficient_evidence",
        })
        builder.add_edge("document_search", "observe")
        builder.add_edge("observe", "decide")
        builder.add_edge("direct_response", "finalize")
        builder.add_edge("grounded_response", "finalize")
        builder.add_edge("insufficient_evidence", "finalize")
        builder.add_edge("finalize", END)
        self.compiled = builder.compile()

    @staticmethod
    def _budget_ms(state: AgentState | None) -> float | None:
        if state is None or "deadline" not in state:
            return None
        return round(max(0, state["deadline"] - perf_counter()) * 1000, 3)

    def _record(self, node: str, started: float, outcome: str = "completed",
                state: AgentState | None = None, decision: AgentDecision | None = None,
                error: Exception | None = None,
                validation_retry_count: int | None = None,
                validation_failure_code: str | None = None) -> AgentNodeTrace:
        duration_ms = round((perf_counter() - started) * 1000, 3)
        budget_ms = self._budget_ms(state)
        error_code = getattr(error, "reason_code", None) if error is not None else None
        if error is not None:
            validation_retry_count = getattr(
                error, "validation_retry_count", validation_retry_count,
            )
            validation_failure_code = getattr(
                error, "validation_failure_code", validation_failure_code,
            )
        self.logger.info("agent_node_completed", extra={
            "agent_node": node, "duration_ms": duration_ms, "outcome": outcome,
            "agent_action": decision.action if decision else None,
            "agent_scope": decision.scope if decision else None,
            "agent_reason": decision.reason_code if decision else None,
            "evidence_count": len(state.get("evidence", [])) if state is not None else None,
            "remaining_budget_ms": budget_ms,
            "agent_error_type": type(error).__name__ if error is not None else None,
            "agent_error_code": error_code,
            "validation_retry_count": validation_retry_count,
            "validation_failure_code": validation_failure_code,
        })
        return AgentNodeTrace(
            node, duration_ms, outcome,
            decision.action if decision else None,
            decision.scope if decision else None,
            decision.reason_code if decision else None,
            len(state.get("evidence", [])) if state is not None else None,
            budget_ms,
            type(error).__name__ if error is not None else None,
            error_code,
            validation_retry_count,
            validation_failure_code,
        )

    def _record_failure(self, state: AgentState, node: str, started: float,
                        error: Exception) -> None:
        failed = self._record(
            node, started, "failed", state, state.get("decision"), error,
        )
        diagnostics = {
            "agent_trace": [*state.get("trace", []), failed],
            "agent_tool_activity": list(state.get("tool_activity", [])),
            "agent_route": state.get("route"),
            "agent_tool_call_count": state.get("tool_call_count", 0),
            "agent_remaining_budget_ms": failed.remaining_budget_ms,
            "agent_error_code": failed.error_code,
        }
        for name, value in diagnostics.items():
            try:
                setattr(error, name, value)
            except (AttributeError, TypeError):
                pass

    @staticmethod
    def _validate_usage(usage: AgentTokenUsage | None) -> None:
        if usage is None:
            return
        if not isinstance(usage, AgentTokenUsage) or any(
            value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ) for value in (usage.prompt_tokens, usage.completion_tokens)
        ):
            raise AgentLLMInvalidResponse("The provider returned invalid token usage.")

    @staticmethod
    def _remaining(state: AgentState) -> float:
        remaining = state["deadline"] - perf_counter()
        if remaining <= 0:
            raise AgentExecutionTimeout("The Agentic execution deadline was reached.")
        return remaining

    def _observations(self, state: AgentState) -> list[dict]:
        searches = [{
            "query": item.query, "result_count": item.result_count, "outcome": item.outcome,
        } for item in state.get("tool_activity", [])]
        evidence = []
        remaining = self.decision_context_bytes
        for index, chunk in enumerate(state.get("evidence", []), 1):
            if remaining <= 0:
                break
            text = chunk.text.encode("utf-8")[:remaining].decode("utf-8", errors="ignore")
            if not text:
                break
            evidence.append({"evidence_id": f"E{index}", "text": text})
            remaining -= len(text.encode("utf-8"))
        return [{"searches": searches, "evidence": evidence}]

    def _decide(self, state: AgentState) -> AgentState:
        started = perf_counter()
        remaining_calls = max(0, self.max_tool_calls - state.get("tool_call_count", 0))
        try:
            decision = self.provider.decide(
                state["question"], self._observations(state), remaining_calls,
                timeout_seconds=self._remaining(state),
            )
            if not isinstance(decision, AgentDecision):
                raise AgentLLMInvalidResponse("The provider returned an invalid Agentic decision.")
            action = decision.action
            if action not in {
                "answer_direct", "search_documents", "answer_grounded", "insufficient_evidence",
            }:
                raise AgentLLMInvalidResponse("The provider returned an unknown Agentic action.")
            allowed_reasons = {
                "answer_direct": {"general_question"},
                "search_documents": {"evidence_needed", "refine_search"},
                "answer_grounded": {"evidence_supports_answer"},
                "insufficient_evidence": {
                    "evidence_does_not_support_answer", "search_budget_exhausted",
                },
            }
            if (decision.reason_code is not None
                    and decision.reason_code not in allowed_reasons[action]):
                raise AgentLLMInvalidResponse(
                    "The provider returned an inconsistent Agentic decision reason."
                )
            self._validate_usage(decision.usage)
            scope = decision.scope or (
                "general_knowledge" if action == "answer_direct" else "document_research"
            )
            if scope == "general_knowledge" and action != "answer_direct":
                raise AgentLLMInvalidResponse(
                    "The provider selected document tools for a general-knowledge decision.",
                )
            if scope == "document_research" and action == "answer_direct":
                raise AgentLLMInvalidResponse(
                    "The provider attempted to answer a document request without evidence.",
                )
            update: AgentState = {"decision": decision}
            if decision.usage is not None:
                update["usages"] = [decision.usage]
            tool_count = state.get("tool_call_count", 0)
            evidence = state.get("evidence", [])
            if action == "answer_direct" and tool_count:
                raise AgentLLMInvalidResponse(
                    "The provider attempted an ungrounded answer after document search.",
                )
            if action == "answer_grounded" and not evidence:
                update["decision"] = AgentDecision(
                    "insufficient_evidence", usage=decision.usage,
                    scope="document_research",
                    reason_code="evidence_does_not_support_answer",
                )
                update["message"] = DEFAULT_INSUFFICIENT
                action = "insufficient_evidence"
            if action == "search_documents":
                if (not isinstance(decision.query, str) or not decision.query.strip()
                        or not isinstance(decision.top_k, int) or isinstance(decision.top_k, bool)
                        or not 1 <= decision.top_k <= self.max_tool_top_k):
                    raise AgentToolInvalid("The model produced invalid document-search arguments.")
                key = _search_identity(decision.query)
                if not key:
                    raise AgentToolInvalid("The model produced an invalid document-search query.")
                if tool_count >= self.max_tool_calls:
                    update["decision"] = AgentDecision(
                        "insufficient_evidence", usage=decision.usage,
                        scope="document_research", reason_code="search_budget_exhausted",
                    )
                    update["message"] = "The document-search call limit was reached without sufficient evidence."
                    action = "insufficient_evidence"
                elif key in state.get("attempted_calls", []):
                    update["decision"] = AgentDecision(
                        "insufficient_evidence", usage=decision.usage,
                        scope="document_research",
                        reason_code="evidence_does_not_support_answer",
                    )
                    update["message"] = "A repeated document search was blocked without sufficient evidence."
                    action = "insufficient_evidence"
            route = "direct" if action == "answer_direct" else "document_research"
            update["route"] = route
        except Exception as exc:
            self._record_failure(state, "decide", started, exc)
            raise
        update["trace"] = [self._record(
            "decide", started, action, state, update["decision"],
        )]
        return update

    @staticmethod
    def _decision_action(state: AgentState) -> AgentAction:
        return cast(AgentAction, state["decision"].action)

    def _document_search(self, state: AgentState) -> AgentState:
        started = perf_counter()
        decision = state["decision"]
        query = cast(str, decision.query)
        top_k = cast(int, decision.top_k)
        key = _search_identity(query)
        try:
            self._remaining(state)
            result = self.search_tool.search(
                query, top_k, state.get("document_ids"), state.get("strategy"),
            )
            self._remaining(state)
        except Exception as exc:
            self._record_failure(state, "document_search", started, exc)
            raise
        existing = {chunk.chunk_id for chunk in state.get("evidence", [])}
        evidence = list(state.get("evidence", []))
        before_count = len(evidence)
        for chunk in result.chunks:
            if chunk.chunk_id not in existing and len(evidence) < self.max_evidence_chunks:
                evidence.append(chunk)
                existing.add(chunk.chunk_id)
        duration_ms = round((perf_counter() - started) * 1000, 3)
        outcome = "results" if result.chunks else "empty"
        activity = AgentToolActivity(
            state.get("tool_call_count", 0) + 1, query, top_k,
            len(result.chunks), duration_ms, outcome,
            _query_fingerprint(query), result.strategy,
            None if state.get("document_ids") is None else len(state["document_ids"]),
            len(evidence) - before_count, self._budget_ms(state),
        )
        self.logger.info("agent_tool_completed", extra={
            "agent_tool": "document_search", "tool_call_count": activity.call_index,
            "retrieved_chunk_count": activity.result_count,
            "duration_ms": duration_ms, "outcome": outcome,
            "query_fingerprint": activity.query_fingerprint,
            "retrieval_strategy": activity.retrieval_strategy,
            "document_filter_count": activity.document_filter_count,
            "evidence_added_count": activity.evidence_added_count,
            "remaining_budget_ms": activity.remaining_budget_ms,
        })
        traced_state: AgentState = {**state, "evidence": evidence}
        return {
            "evidence": evidence,
            "attempted_calls": [*state.get("attempted_calls", []), key],
            "tool_call_count": activity.call_index,
            "tool_activity": [activity],
            "trace": [self._record(
                "document_search", started, outcome, traced_state, decision,
            )],
        }

    def _observe(self, state: AgentState) -> AgentState:
        started = perf_counter()
        self._remaining(state)
        return {"trace": [self._record(
            "observe", started,
            "evidence_available" if state.get("evidence") else "empty", state,
        )]}

    def _direct_response(self, state: AgentState) -> AgentState:
        started = perf_counter()
        try:
            generation = self.provider.generate(
                DIRECT_SYSTEM_PROMPT, state["question"],
                timeout_seconds=self._remaining(state),
            )
            if (not isinstance(generation, AgentGeneration)
                    or not isinstance(generation.answer, str)
                    or not generation.answer.strip() or len(generation.answer) > 12000):
                raise AgentLLMInvalidResponse("The provider returned an invalid Agentic response.")
            self._validate_usage(generation.usage)
            self._remaining(state)
        except Exception as exc:
            self._record_failure(state, "direct_response", started, exc)
            raise
        update: AgentState = {
            "status": "direct", "answer": generation.answer.strip(), "message": None,
            "citations": [], "trace": [self._record(
                "direct_response", started, "completed", state,
            )],
        }
        if generation.usage is not None:
            update["usages"] = [generation.usage]
        return update

    def _grounded_response(self, state: AgentState) -> AgentState:
        started = perf_counter()
        try:
            grounded = self.grounding.answer(
                state["question"], state.get("evidence", []),
                timeout_seconds=self._remaining(state),
            )
            self._remaining(state)
        except Exception as exc:
            self._record_failure(state, "grounded_response", started, exc)
            raise
        status = "grounded" if grounded.status == "answered" else "insufficient_evidence"
        update: AgentState = {
            "status": status, "answer": grounded.answer, "message": grounded.message,
            "citations": grounded.citations,
            "trace": [self._record(
                "grounded_response", started, status, state, state.get("decision"),
                validation_retry_count=grounded.validation_retry_count,
                validation_failure_code=grounded.validation_failure_code,
            )],
        }
        if grounded.usage is not None:
            update["usages"] = [AgentTokenUsage(
                grounded.usage.prompt_tokens, grounded.usage.completion_tokens,
            )]
        return update

    def _insufficient_evidence(self, state: AgentState) -> AgentState:
        started = perf_counter()
        return {
            "status": "insufficient_evidence", "answer": None,
            "message": state.get("message") or DEFAULT_INSUFFICIENT,
            "citations": [],
            "trace": [self._record(
                "insufficient_evidence", started, "completed", state,
                state.get("decision"),
            )],
        }

    def _finalize(self, state: AgentState) -> AgentState:
        started = perf_counter()
        self._remaining(state)
        return {"trace": [self._record("finalize", started, state["status"], state)]}

    def invoke(self, question: str, document_ids: list[UUID] | None,
               strategy: RetrievalStrategy | None, step_limit: int,
               execution_timeout_seconds: float) -> AgentState:
        return cast(AgentState, self.compiled.invoke({
            "question": question, "document_ids": document_ids, "strategy": strategy,
            "deadline": perf_counter() + execution_timeout_seconds,
            "evidence": [], "attempted_calls": [], "tool_call_count": 0,
            "trace": [], "tool_activity": [], "usages": [],
        }, config={"recursion_limit": step_limit}))
