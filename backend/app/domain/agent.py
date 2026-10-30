"""Framework-free contracts for bounded tool-using Agentic execution."""
from dataclasses import dataclass
from typing import Literal

from app.domain.rag import Citation


AgentAction = Literal[
    "answer_direct", "search_documents", "answer_grounded", "insufficient_evidence",
]
AgentScope = Literal["general_knowledge", "document_research"]
AgentRoute = Literal["direct", "document_research"]
AgentStatus = Literal["direct", "grounded", "insufficient_evidence"]


@dataclass(frozen=True)
class AgentTokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class AgentDecision:
    action: AgentAction
    query: str | None = None
    top_k: int | None = None
    usage: AgentTokenUsage | None = None
    scope: AgentScope | None = None


@dataclass(frozen=True)
class AgentGeneration:
    answer: str
    usage: AgentTokenUsage | None = None


@dataclass(frozen=True)
class AgentNodeTrace:
    node: str
    duration_ms: float
    outcome: str


@dataclass(frozen=True)
class AgentToolActivity:
    call_index: int
    query: str
    top_k: int
    result_count: int
    duration_ms: float
    outcome: Literal["results", "empty"]


@dataclass(frozen=True)
class AgentExecutionMetadata:
    route: AgentRoute
    duration_ms: float
    steps: int
    tool_call_count: int
    node_trace: list[AgentNodeTrace]
    tool_activity: list[AgentToolActivity]
    usage: AgentTokenUsage | None


@dataclass(frozen=True)
class AgentResult:
    status: AgentStatus
    answer: str | None
    message: str | None
    citations: list[Citation]
    provider: str
    model: str
    document_tools_available: Literal[True]
    metadata: AgentExecutionMetadata | None


class AgentQuestionInvalid(Exception):
    pass


class AgentToolInvalid(Exception):
    pass


class AgentLLMUnavailable(Exception):
    pass


class AgentLLMTimeout(Exception):
    pass


class AgentLLMInvalidResponse(Exception):
    pass


class AgentExecutionLimit(Exception):
    pass


class AgentExecutionTimeout(Exception):
    pass
