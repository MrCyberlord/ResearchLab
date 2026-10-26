"""Small contracts for later use cases; no providers are invoked in Milestone 1."""
from typing import Protocol
from uuid import UUID

from app.domain.models import RunSummary


class LLMProvider(Protocol):
    async def generate(self, prompt: str) -> str: ...


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class RunRepository(Protocol):
    async def get(self, run_id: UUID) -> RunSummary | None: ...
    async def save(self, run: RunSummary) -> None: ...
