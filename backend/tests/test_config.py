import pytest
from pydantic import ValidationError
from app.infrastructure.config import Settings, get_settings


def test_defaults():
    settings = Settings(_env_file=None, database_url=None)
    assert settings.app_env == "development"
    assert settings.cors_origins == ["http://localhost:5173"]
    assert settings.database_url is None


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("CORS_ORIGINS", '["http://localhost:3000"]')
    monkeypatch.setenv("DATABASE_URL", "")
    settings = Settings(_env_file=None)
    assert settings.app_env == "test"
    assert settings.cors_origins == ["http://localhost:3000"]
    assert settings.database_url is None


@pytest.mark.parametrize("values", [
    {"app_env": "invalid"}, {"log_level": "invalid"},
    {"database_url": "sqlite:///data.db"},
    {"database_url": "postgresql://localhost/researchlab"},
    {"cors_origins": ["*"]}, {"cors_origins": ["https://example.com/path"]},
])
def test_invalid_settings(values):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)


def test_database_secret_is_redacted():
    settings = Settings(_env_file=None, database_url="postgresql+psycopg://localhost/researchlab")
    assert settings.database_url is not None
    assert "postgresql+psycopg" not in repr(settings)


def test_env_file_precedence(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("APP_ENV=test\nLOG_LEVEL=WARNING\n", encoding="utf-8")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    assert Settings(_env_file=env).app_env == "test"
    assert Settings(_env_file=env).log_level == "ERROR"


def test_cache_reset(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("APP_ENV", "test")
    assert get_settings().app_env == "test"
    monkeypatch.setenv("APP_ENV", "development")
    assert get_settings().app_env == "test"
    get_settings.cache_clear()
    assert get_settings().app_env == "development"
    get_settings.cache_clear()
