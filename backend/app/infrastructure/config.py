from functools import lru_cache
from app.domain.embeddings import MINILM_MODEL, MINILM_REVISION
from pathlib import Path
import re
from typing import Literal

from pydantic import PostgresDsn, SecretStr, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[3] / ".env",
        env_file_encoding="utf-8", extra="ignore",
    )
    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    cors_origins: list[str] = ["http://localhost:5173"]
    database_url: SecretStr | None = None
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1, le=50 * 1024 * 1024)
    max_chunks: int = Field(default=5000, ge=1, le=10000)
    pdf_max_pages: int = Field(default=200, ge=1, le=1000)
    pdf_max_chars: int = Field(default=2_000_000, ge=1, le=5_000_000)
    pdf_timeout_seconds: float = Field(default=15, ge=0.1, le=60)
    pdf_memory_mb: int = Field(default=256, ge=64, le=1024)
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=150, ge=0)
    embedding_provider: Literal["local", "sentence_transformers"] = "sentence_transformers"
    embedding_model: Literal["local-hash-v1", "sentence-transformers/all-MiniLM-L6-v2"] = MINILM_MODEL
    embedding_revision: str = MINILM_REVISION
    embedding_cache_dir: Path = Path(__file__).resolve().parents[3] / ".cache" / "models"
    embedding_local_files_only: bool = False
    embedding_batch_size: int = Field(default=16, ge=1, le=128)
    embedding_cpu_threads: int = Field(default=2, ge=1, le=32)
    embedding_dimension: Literal[384] = 384
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_min_similarity: float = Field(default=0.25, ge=-1, le=1)
    retrieval_strategy: Literal["semantic", "hybrid", "hybrid_rerank"] = "semantic"
    semantic_candidate_count: int = Field(default=20, ge=20, le=100)
    keyword_candidate_count: int = Field(default=20, ge=20, le=100)
    rrf_fusion_constant: int = Field(default=60, ge=1, le=1000)
    reranker_provider: Literal["cross_encoder"] = "cross_encoder"
    reranker_model: Literal["cross-encoder/ms-marco-MiniLM-L6-v2"] = RERANKER_MODEL
    reranker_revision: str = RERANKER_REVISION
    reranker_cache_dir: Path = Path(__file__).resolve().parents[3] / ".cache" / "models"
    reranker_local_files_only: bool = False
    reranker_batch_size: int = Field(default=16, ge=1, le=64)
    reranker_cpu_threads: int = Field(default=2, ge=1, le=32)
    reranker_candidate_count: int = Field(default=20, ge=20, le=100)
    rag_max_context_bytes: int = Field(default=4000, ge=128, le=50000)
    rag_max_prompt_bytes: int = Field(default=6000, ge=2048, le=64000)
    llm_provider: Literal["ollama"] = "ollama"
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = Field(default="gemma3:4b", min_length=1, max_length=120)
    llm_timeout_seconds: float = Field(default=120, ge=1, le=600)
    llm_context_tokens: int = Field(default=8192, ge=2048, le=65536)
    llm_max_output_tokens: int = Field(default=512, ge=64, le=4096)
    llm_max_response_bytes: int = Field(default=65536, ge=1024, le=1048576)
    agent_graph_step_limit: int = Field(default=8, ge=4, le=20)
    agent_execution_timeout_seconds: float = Field(default=120, ge=1, le=600)
    agent_max_question_chars: int = Field(default=2000, ge=100, le=10000)
    agent_max_output_tokens: int = Field(default=384, ge=64, le=4096)
    agent_max_response_bytes: int = Field(default=65536, ge=1024, le=1048576)

    @model_validator(mode="after")
    def validate_chunk_config(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        expected = "local-hash-v1" if self.embedding_provider == "local" else MINILM_MODEL
        if self.embedding_model != expected:
            raise ValueError("EMBEDDING_MODEL does not match EMBEDDING_PROVIDER")
        if self.embedding_provider == "sentence_transformers":
            if not re.fullmatch(r"[0-9a-f]{40}", self.embedding_revision):
                raise ValueError("EMBEDDING_REVISION must be an immutable 40-character model commit")
        if not re.fullmatch(r"[0-9a-f]{40}", self.reranker_revision):
            raise ValueError("RERANKER_REVISION must be an immutable 40-character model commit")
        if self.rag_max_context_bytes > self.rag_max_prompt_bytes:
            raise ValueError("RAG_MAX_CONTEXT_BYTES must not exceed RAG_MAX_PROMPT_BYTES")
        if self.rag_max_prompt_bytes + self.llm_max_output_tokens + 1024 > self.llm_context_tokens:
            raise ValueError("Prompt bytes, output tokens and safety margin exceed LLM_CONTEXT_TOKENS")
        if self.agent_max_output_tokens + 1024 > self.llm_context_tokens:
            raise ValueError("Agent output tokens and safety margin exceed LLM_CONTEXT_TOKENS")
        return self

    @field_validator("llm_base_url")
    @classmethod
    def validate_llm_base_url(cls, value):
        from urllib.parse import urlsplit
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("LLM_BASE_URL must be HTTP(S), without credentials, query or fragment")
        parsed.port  # Reject invalid port values.
        return value.rstrip("/")

    @field_validator("llm_model")
    @classmethod
    def validate_llm_model(cls, value):
        if not value.strip() or any(ord(char) < 32 for char in value):
            raise ValueError("LLM_MODEL must be a nonempty model name without control characters")
        return value.strip()

    @field_validator("embedding_cache_dir", mode="before")
    @classmethod
    def resolve_embedding_cache(cls, value):
        path = Path(value)
        return path if path.is_absolute() else Path(__file__).resolve().parents[3] / path

    @field_validator("reranker_cache_dir", mode="before")
    @classmethod
    def resolve_reranker_cache(cls, value):
        path = Path(value)
        return path if path.is_absolute() else Path(__file__).resolve().parents[3] / path

    @field_validator("database_url", mode="before")
    @classmethod
    def validate_database_url(cls, value):
        if value is None or value == "":
            return None
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        url = PostgresDsn(raw)
        if url.scheme != "postgresql+psycopg":
            raise ValueError("Use postgresql+psycopg for DATABASE_URL")
        return SecretStr(raw)

    @field_validator("cors_origins")
    @classmethod
    def explicit_origins(cls, origins):
        from urllib.parse import urlsplit
        for origin in origins:
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or parsed.username:
                raise ValueError("CORS origins must be explicit HTTP(S) origins without paths")
        return origins


@lru_cache
def get_settings() -> Settings:
    return Settings()
