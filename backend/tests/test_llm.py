"""No external LLM calls: transport-level tests of the real Ollama adapter."""
import json

import httpx
import pytest

from app.domain.rag import LLMInvalidResponse, LLMTimeout, LLMUnavailable
from app.infrastructure.llm import OllamaLLM, create_llm_provider
from app.infrastructure.config import Settings


def response_payload(**overrides):
    result = {
        "model": "gemma3:4b", "done": True, "done_reason": "stop",
        "message": {"role": "assistant", "content": json.dumps({
            "answer": "The warranty lasts three years [S1].",
            "evidence": [{"source_id": "S1", "quote": "The warranty lasts three years."}],
            "insufficient_evidence": False,
        })},
        "prompt_eval_count": 50, "eval_count": 22,
    }
    result.update(overrides)
    return result


def provider(handler, **options):
    return OllamaLLM("http://127.0.0.1:11434", "gemma3:4b",
                     transport=httpx.MockTransport(handler), **options)


def test_ollama_request_and_usage_are_real_contract():
    captured = {}
    def handler(request):
        captured["request"] = request
        return httpx.Response(200, json=response_payload())
    result = provider(handler).generate("System instructions", "Question and sources", ["S1"])
    request = captured["request"]
    body = json.loads(request.content)
    assert request.url == "http://127.0.0.1:11434/api/chat"
    assert body["model"] == "gemma3:4b" and body["stream"] is False
    assert body["messages"] == [{"role": "system", "content": "System instructions"},
                                 {"role": "user", "content": "Question and sources"}]
    assert body["options"] == {"temperature": 0, "num_ctx": 8192, "num_predict": 512}
    assert body["format"]["$defs"]["_Evidence"]["properties"]["source_id"]["enum"] == ["S1"]
    assert result.answer == "The warranty lasts three years [S1]."
    assert result.evidence[0].source_id == "S1"
    assert result.usage.prompt_tokens == 50 and result.usage.completion_tokens == 22


def test_provider_factory_uses_configuration():
    settings = Settings(_env_file=None, llm_model="configured-model", llm_base_url="http://localhost:11435")
    llm = create_llm_provider(settings)
    assert llm.model == "configured-model" and llm.base_url == "http://localhost:11435"


@pytest.mark.parametrize("status", [301, 400, 401, 404, 429, 500, 503])
def test_provider_http_errors_are_safe(status):
    llm = provider(lambda request: httpx.Response(status, text="secret upstream details"))
    with pytest.raises(LLMUnavailable) as error:
        llm.generate("system", "user", ["S1"])
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("failure,expected", [
    (httpx.ReadTimeout, LLMTimeout), (httpx.ConnectTimeout, LLMTimeout),
    (httpx.ConnectError, LLMUnavailable), (httpx.ReadError, LLMUnavailable),
])
def test_provider_network_failures(failure, expected):
    def handler(request): raise failure("secret connection details", request=request)
    with pytest.raises(expected) as error:
        provider(handler).generate("system", "user", ["S1"])
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("change", [
    {"done": False}, {"done": "true"}, {"done_reason": "length"},
    {"message": {"role": "user", "content": "{}"}},
    {"message": {"role": "assistant", "content": "not json"}},
    {"message": {"role": "assistant", "content": "```json\n{}\n```"}},
    {"prompt_eval_count": -1}, {"eval_count": True}, {"eval_count": "12"},
])
def test_malformed_envelopes_are_rejected(change):
    llm = provider(lambda request: httpx.Response(200, json=response_payload(**change)))
    with pytest.raises(LLMInvalidResponse): llm.generate("system", "user", ["S1"])


@pytest.mark.parametrize("answer", [
    {"answer": "unsupported", "evidence": [], "insufficient_evidence": False},
    {"answer": "", "evidence": [{"source_id": "S1", "quote": "some quote"}], "insufficient_evidence": False},
    {"answer": "unsupported", "evidence": [{"source_id": "S999", "quote": "some quote"}], "insufficient_evidence": False},
    {"answer": "unsupported", "evidence": [{"source_id": "S1", "quote": "   "}], "insufficient_evidence": False},
    {"answer": "unsupported", "evidence": [{"source_id": "S1", "quote": "x" * 1001}], "insufficient_evidence": False},
    {"answer": "unsupported", "evidence": [{"source_id": "S1", "quote": "some quote"}], "insufficient_evidence": "false"},
    {"answer": "unsupported", "evidence": [{"source_id": "S1", "quote": "some quote"}], "insufficient_evidence": False, "extra": "ignored?"},
])
def test_invalid_structured_answers_are_rejected(answer):
    payload = response_payload(message={"role": "assistant", "content": json.dumps(answer)})
    llm = provider(lambda request: httpx.Response(200, json=payload))
    with pytest.raises(LLMInvalidResponse): llm.generate("system", "user", ["S1"])


def test_insufficient_evidence_is_explicit_and_usage_can_be_absent():
    payload = response_payload(message={"role": "assistant", "content": json.dumps({
        "answer": "", "evidence": [], "insufficient_evidence": True})})
    payload.pop("prompt_eval_count"); payload.pop("eval_count")
    result = provider(lambda request: httpx.Response(200, json=payload)).generate("system", "user", ["S1"])
    assert result.insufficient_evidence and result.answer == "" and result.evidence == [] and result.usage is None


def test_response_limit_with_and_without_length():
    for headers in ({"content-length": "500"}, {}):
        def handler(request):
            return httpx.Response(200, stream=httpx.ByteStream(b"x" * 500), headers=headers)
        with pytest.raises(LLMInvalidResponse):
            provider(handler, max_response_bytes=100).generate("system", "user", ["S1"])


def test_unexpected_encoding_and_invalid_utf8_are_rejected():
    for response in [httpx.Response(200, content=b"not gzip", headers={"content-encoding": "identity-custom"}),
                     httpx.Response(200, content=b"\xff")]:
        with pytest.raises(LLMInvalidResponse):
            provider(lambda request: response).generate("system", "user", ["S1"])


def test_duplicate_json_keys_are_rejected():
    payload = b'{"done":true,"done":false}'
    with pytest.raises(LLMInvalidResponse):
        provider(lambda request: httpx.Response(200, content=payload)).generate("system", "user", ["S1"])


@pytest.mark.parametrize("source_ids", [[], ["S1", "S1"], [""], [{}]])
def test_invalid_context_labels_do_not_call_provider(source_ids):
    def handler(request): raise AssertionError("No HTTP call expected")
    with pytest.raises(LLMInvalidResponse): provider(handler).generate("system", "user", source_ids)
