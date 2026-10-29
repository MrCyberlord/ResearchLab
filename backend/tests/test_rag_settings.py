import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.infrastructure.config import Settings
from app.main import create_app


@pytest.mark.parametrize("values", [
    {"llm_provider": "unsupported"},
    {"llm_base_url": "file:///model"},
    {"llm_base_url": "http://user:secret@example.test"},
    {"llm_base_url": "http://localhost:11434?api_key=secret"},
    {"llm_base_url": "http://localhost:bad"},
    {"llm_model": " "},
    {"llm_timeout_seconds": 0},
    {"rag_max_context_bytes": 7000},
    {"llm_context_tokens": 4096},
    {"retrieval_min_similarity": float("nan")},
    {"retrieval_top_k": 21},
    {"retrieval_strategy": "keyword"},
    {"semantic_candidate_count": 19},
    {"keyword_candidate_count": 101},
    {"rrf_fusion_constant": 0},
    {"reranker_provider": "remote"},
    {"reranker_model": "untrusted/model"},
    {"reranker_revision": "main"},
    {"reranker_candidate_count": 19},
])
def test_rag_settings_reject_unsafe_or_inconsistent_values(values):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)


def test_local_provider_defaults_and_url_normalization():
    settings = Settings(_env_file=None, llm_base_url="http://127.0.0.1:11434/")
    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "gemma3:4b"
    assert settings.llm_base_url == "http://127.0.0.1:11434"
    assert settings.retrieval_strategy == "semantic"
    assert settings.semantic_candidate_count == settings.keyword_candidate_count == 20
    assert settings.rrf_fusion_constant == 60
    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert len(settings.reranker_revision) == 40
    assert settings.rag_max_prompt_bytes + settings.llm_max_output_tokens + 1024 <= settings.llm_context_tokens


@pytest.mark.parametrize("path", ["/api/retrieval", "/api/rag/questions"])
def test_rag_json_request_body_is_bounded_before_validation(path):
    with TestClient(create_app(Settings(_env_file=None, database_url=None))) as client:
        response = client.post(path, content=b"x" * 32769, headers={"Content-Type": "application/json", "Origin": "http://localhost:5173"})
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "request_too_large"
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
