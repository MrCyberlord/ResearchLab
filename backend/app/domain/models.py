from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class ExecutionMode(StrEnum):
    RAG = "rag"
    AGENTIC = "agentic"


@dataclass(frozen=True)
class RunSummary:
    id: UUID
    mode: ExecutionMode
