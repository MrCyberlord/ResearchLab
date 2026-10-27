from functools import lru_cache
from app.domain.embeddings import MINILM_MODEL, MINILM_REVISION
from pathlib import Path
from typing import Literal

from pydantic import PostgresDsn, SecretStr, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    @model_validator(mode="after")
    def validate_chunk_config(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        expected = "local-hash-v1" if self.embedding_provider == "local" else MINILM_MODEL
        if self.embedding_model != expected:
            raise ValueError("EMBEDDING_MODEL does not match EMBEDDING_PROVIDER")
        if self.embedding_provider == "sentence_transformers":
            import re
            if not re.fullmatch(r"[0-9a-f]{40}", self.embedding_revision):
                raise ValueError("EMBEDDING_REVISION must be an immutable 40-character model commit")
        return self

    @field_validator("embedding_cache_dir", mode="before")
    @classmethod
    def resolve_embedding_cache(cls, value):
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
