"""Adapter contracts consumed by application services."""
from typing import Protocol
from contextlib import AbstractContextManager
from app.domain.documents import Chunk, Document, PageText
from uuid import UUID

from app.domain.models import RunSummary
from app.domain.rag import LLMResult, RetrievedChunk
from app.domain.agent import AgentGeneration


class LLMProvider(Protocol):
    provider: str
    model: str

    def generate(self, system_prompt: str, user_prompt: str, source_ids: list[str]) -> LLMResult: ...


class AgentGenerationProvider(Protocol):
    """Ungrounded text generation boundary used by the Agentic workflow only."""
    provider: str
    model: str

    def generate(self, system_prompt: str, user_prompt: str) -> AgentGeneration: ...


class EmbeddingProvider(Protocol):
    model: str
    dimension: int
    revision: str
    profile: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class RerankerProvider(Protocol):
    provider: str
    model: str
    revision: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class RunRepository(Protocol):
    async def get(self, run_id: UUID) -> RunSummary | None: ...
    async def save(self, run: RunSummary) -> None: ...


class PDFExtractor(Protocol):
    def extract(self, data: bytes) -> list[PageText]: ...


class DocumentRepository(Protocol):
    def transaction(self) -> AbstractContextManager: ...
    def check_profile(self, profile: str) -> None: ...
    def bind_profile(self, profile: str) -> None: ...
    def find_by_hash(self, content_hash: str) -> Document | None: ...
    def add(self, document: Document, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...


class RetrievalRepository(Protocol):
    def check_profile(self, profile: str) -> None: ...
    def search(self, vector: list[float], profile: str, top_k: int, min_similarity: float,
               document_ids: list[UUID] | None) -> list[RetrievedChunk]: ...
    def keyword_search(self, query: str, vector: list[float], profile: str, top_k: int,
                       document_ids: list[UUID] | None) -> list[RetrievedChunk]: ...
    def list_documents(self, limit: int, offset: int) -> tuple[list[Document], int]: ...
