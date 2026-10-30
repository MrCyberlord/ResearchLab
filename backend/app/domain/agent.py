"""Framework-free contracts for the bounded Milestone 3A agent workflow."""
from dataclasses import dataclass
from typing import Literal


AgentRoute = Literal["direct", "document_tools_unavailable"]


@dataclass(frozen=True)
class AgentTokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


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
class AgentExecutionMetadata:
    route: AgentRoute
    duration_ms: float
    steps: int
    node_trace: list[AgentNodeTrace]
    usage: AgentTokenUsage | None


@dataclass(frozen=True)
class AgentResult:
    answer: str
    provider: str
    model: str
    document_tools_available: Literal[False]
    metadata: AgentExecutionMetadata | None


class AgentQuestionInvalid(Exception):
    pass


class AgentLLMUnavailable(Exception):
    pass


class AgentLLMTimeout(Exception):
    pass


class AgentLLMInvalidResponse(Exception):
    pass


class AgentExecutionLimit(Exception):
    pass
