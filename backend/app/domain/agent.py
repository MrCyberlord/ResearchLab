"""Framework-free contracts for bounded tool-using Agentic execution."""
from dataclasses import dataclass
from typing import Literal

from app.domain.rag import Citation, RetrievalStrategy


AgentAction = Literal[
    "answer_direct", "search_documents", "answer_grounded", "insufficient_evidence",
]
AgentScope = Literal["general_knowledge", "document_research"]
AgentRoute = Literal["direct", "document_research"]
AgentStatus = Literal["direct", "grounded", "insufficient_evidence"]
AgentDecisionReason = Literal[
    "general_question", "evidence_needed", "refine_search",
    "evidence_supports_answer", "evidence_does_not_support_answer",
    "search_budget_exhausted",
]


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
    reason_code: AgentDecisionReason | None = None


@dataclass(frozen=True)
class AgentGeneration:
    answer: str
    usage: AgentTokenUsage | None = None


@dataclass(frozen=True)
class AgentNodeTrace:
    node: str
    duration_ms: float
    outcome: str
    decision_action: AgentAction | None = None
    decision_scope: AgentScope | None = None
    decision_reason: AgentDecisionReason | None = None
    evidence_count: int | None = None
    remaining_budget_ms: float | None = None
    error_type: str | None = None
    error_code: str | None = None
    validation_retry_count: int | None = None
    validation_failure_code: str | None = None


@dataclass(frozen=True)
class AgentToolActivity:
    call_index: int
    query: str
    top_k: int
    result_count: int
    duration_ms: float
    outcome: Literal["results", "empty"]
    query_fingerprint: str | None = None
    retrieval_strategy: RetrievalStrategy | None = None
    document_filter_count: int | None = None
    evidence_added_count: int | None = None
    remaining_budget_ms: float | None = None


@dataclass(frozen=True)
class AgentExecutionMetadata:
    route: AgentRoute
    duration_ms: float
    steps: int
    tool_call_count: int
    node_trace: list[AgentNodeTrace]
    tool_activity: list[AgentToolActivity]
    usage: AgentTokenUsage | None
    execution_budget_ms: float | None = None
    remaining_budget_ms: float | None = None


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
