"""Bounded LangGraph orchestration for the Milestone 3A agent foundation."""
from operator import add
import logging
from time import perf_counter
from typing import Annotated, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from app.application.ports import AgentGenerationProvider
from app.domain.agent import AgentGeneration, AgentLLMInvalidResponse, AgentNodeTrace, AgentRoute


class AgentState(TypedDict, total=False):
    question: str
    route: AgentRoute
    generation: AgentGeneration
    answer: str
    trace: Annotated[list[AgentNodeTrace], add]


_DOCUMENT_REQUEST_MARKERS = (
    "uploaded document", "uploaded pdf", "my document", "my pdf",
    "selected document", "selected pdf", "these documents", "these files",
    "search the document", "search my files", "according to the document",
    "according to the pdf", "in the pdf", "from the pdf",
)


def choose_route(question: str) -> AgentRoute:
    normalized = " ".join(question.casefold().split())
    return ("document_tools_unavailable"
            if any(marker in normalized for marker in _DOCUMENT_REQUEST_MARKERS)
            else "direct")


class AgentGraph:
    """One routing decision and one generation call; no tool loop or memory."""

    def __init__(self, provider: AgentGenerationProvider):
        self.provider = provider
        self.logger = logging.getLogger("researchlab")
        builder = StateGraph(AgentState)
        builder.add_node("route", self._route)
        builder.add_node("direct_response", self._direct_response)
        builder.add_node("document_tools_unavailable_response", self._document_tools_unavailable_response)
        builder.add_node("finalize", self._finalize)
        builder.add_edge(START, "route")
        builder.add_conditional_edges(
            "route", self._route_key,
            {"direct": "direct_response",
             "document_tools_unavailable": "document_tools_unavailable_response"},
        )
        builder.add_edge("direct_response", "finalize")
        builder.add_edge("document_tools_unavailable_response", "finalize")
        builder.add_edge("finalize", END)
        self.compiled = builder.compile()

    def _record(self, node: str, started: float, outcome: str = "completed") -> AgentNodeTrace:
        duration_ms = round((perf_counter() - started) * 1000, 3)
        self.logger.info("agent_node_completed", extra={
            "agent_node": node, "duration_ms": duration_ms, "outcome": outcome,
        })
        return AgentNodeTrace(node, duration_ms, outcome)

    def _route(self, state: AgentState) -> AgentState:
        started = perf_counter()
        route = choose_route(state["question"])
        return {"route": route, "trace": [self._record("route", started, route)]}

    @staticmethod
    def _route_key(state: AgentState) -> AgentRoute:
        return cast(AgentRoute, state["route"])

    def _generate(self, state: AgentState, node: str, system_prompt: str) -> AgentState:
        started = perf_counter()
        try:
            generation = self.provider.generate(system_prompt, state["question"])
            if (not isinstance(generation, AgentGeneration)
                    or not isinstance(generation.answer, str)
                    or not generation.answer.strip()
                    or len(generation.answer) > 12000):
                raise AgentLLMInvalidResponse("The provider returned an invalid Agentic response.")
        except Exception:
            self._record(node, started, "failed")
            raise
        return {"generation": generation, "trace": [self._record(node, started)]}

    def _direct_response(self, state: AgentState) -> AgentState:
        return self._generate(state, "direct_response", (
            "You are the ResearchLab Agentic foundation. Answer the user's general question "
            "clearly and concisely. You have no tools, browsing, document access, persistent memory, "
            "or citations in this milestone. Never imply that you inspected files or external sources."
        ))

    def _document_tools_unavailable_response(self, state: AgentState) -> AgentState:
        return self._generate(state, "document_tools_unavailable_response", (
            "You are the ResearchLab Agentic foundation. The user is asking about documents, but "
            "document-search tools are not available in this workflow yet. State that limitation "
            "plainly. You may offer general next steps, but do not claim to inspect a file, invent "
            "facts, grounding, quotes, or citations."
        ))

    def _finalize(self, state: AgentState) -> AgentState:
        started = perf_counter()
        generation = state["generation"]
        return {"answer": generation.answer, "trace": [self._record("finalize", started)]}

    def invoke(self, question: str, step_limit: int) -> AgentState:
        return cast(AgentState, self.compiled.invoke(
            {"question": question, "trace": []},
            config={"recursion_limit": step_limit},
        ))
