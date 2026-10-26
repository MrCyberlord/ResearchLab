from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import PostgresDsn, SecretStr, field_validator
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
