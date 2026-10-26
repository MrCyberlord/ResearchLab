import json
import logging
from fastapi.testclient import TestClient
from app.main import create_app
from app.infrastructure.config import Settings
from app.infrastructure.logging import JsonFormatter


def test_health_without_database():
    with TestClient(create_app(Settings(_env_file=None, database_url=None))) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "researchlab-api"}
        assert client.app.state.session_factory is None
        assert client.post("/api/rag").status_code == 404
        assert client.post("/api/agentic").status_code == 404


def test_cors():
    with TestClient(create_app(Settings(_env_file=None, database_url=None))) as client:
        response = client.options("/api/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
        response = client.get("/api/health", headers={"Origin": "https://unknown.example"})
        assert "access-control-allow-origin" not in response.headers


def test_json_logging():
    record = logging.LogRecord("researchlab", logging.INFO, __file__, 1, "request_completed", (), None)
    record.status_code = 200
    payload = json.loads(JsonFormatter().format(record))
    assert payload["event"] == "request_completed"
    assert payload["status_code"] == 200
    assert payload["level"] == "INFO"
