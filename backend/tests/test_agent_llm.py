"""Transport tests for the real Ollama Agentic adapter; no external calls."""
import json

import httpx
import pytest

from app.domain.agent import AgentLLMInvalidResponse, AgentLLMTimeout, AgentLLMUnavailable
from app.infrastructure.agent_llm import OllamaAgentLLM, create_agent_generation_provider
from app.infrastructure.config import Settings


def payload(**overrides):
    result = {
        "model": "gemma3:4b", "done": True, "done_reason": "stop",
        "message": {"role": "assistant", "content": json.dumps({"answer": "Vectors encode features."})},
        "prompt_eval_count": 15, "eval_count": 7,
    }
    result.update(overrides)
    return result


def provider(handler, **options):
    return OllamaAgentLLM("http://127.0.0.1:11434", "gemma3:4b",
                          transport=httpx.MockTransport(handler), **options)


def test_agent_ollama_request_and_structured_response_contract():
    captured = {}
    def handler(request):
        captured["request"] = request
        return httpx.Response(200, json=payload())
    result = provider(handler).generate("No tools.", "Explain vectors.")
    body = json.loads(captured["request"].content)
    assert captured["request"].url == "http://127.0.0.1:11434/api/chat"
    assert body["messages"] == [{"role": "system", "content": "No tools."},
                                {"role": "user", "content": "Explain vectors."}]
    assert body["stream"] is False and body["options"]["temperature"] == 0
    assert body["format"]["properties"]["answer"]
    assert result.answer == "Vectors encode features."
    assert result.usage.prompt_tokens == 15 and result.usage.completion_tokens == 7


def test_agent_provider_factory_uses_agent_limits():
    settings = Settings(_env_file=None, llm_model="configured", agent_execution_timeout_seconds=33,
                        agent_max_output_tokens=256, agent_max_response_bytes=4096)
    result = create_agent_generation_provider(settings)
    assert result.model == "configured" and result.timeout_seconds == 33
    assert result.max_output_tokens == 256 and result.max_response_bytes == 4096


@pytest.mark.parametrize("status", [301, 400, 404, 429, 500, 503])
def test_http_failures_are_safe(status):
    with pytest.raises(AgentLLMUnavailable) as error:
        provider(lambda request: httpx.Response(status, text="secret upstream details")).generate("system", "user")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("failure,expected", [
    (httpx.ReadTimeout, AgentLLMTimeout), (httpx.ConnectTimeout, AgentLLMTimeout),
    (httpx.ConnectError, AgentLLMUnavailable), (httpx.ReadError, AgentLLMUnavailable),
])
def test_network_failures_are_mapped(failure, expected):
    def handler(request):
        raise failure("private network detail", request=request)
    with pytest.raises(expected):
        provider(handler).generate("system", "user")


@pytest.mark.parametrize("change", [
    {"done": False}, {"done_reason": "length"},
    {"message": {"role": "user", "content": "{}"}},
    {"message": {"role": "assistant", "content": "not-json"}},
    {"message": {"role": "assistant", "content": json.dumps({"answer": " "})}},
    {"message": {"role": "assistant", "content": json.dumps({"answer": "ok", "extra": True})}},
    {"eval_count": -1},
])
def test_invalid_envelopes_or_answers_are_rejected(change):
    with pytest.raises(AgentLLMInvalidResponse):
        provider(lambda request: httpx.Response(200, json=payload(**change))).generate("system", "user")


def test_usage_can_be_absent_and_response_size_is_bounded():
    without_usage = payload()
    without_usage.pop("prompt_eval_count"); without_usage.pop("eval_count")
    result = provider(lambda request: httpx.Response(200, json=without_usage)).generate("system", "user")
    assert result.usage is None
    with pytest.raises(AgentLLMInvalidResponse):
        provider(lambda request: httpx.Response(200, content=b"x" * 500),
                 max_response_bytes=100).generate("system", "user")
