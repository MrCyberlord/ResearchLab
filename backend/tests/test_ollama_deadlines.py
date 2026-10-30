"""Deterministic failure injection for absolute Ollama client deadlines."""
import asyncio
import json
from time import perf_counter

import httpx
import pytest

from app.domain.agent import AgentLLMTimeout, AgentLLMUnavailable
from app.domain.rag import LLMTimeout
from app.infrastructure.agent_llm import OllamaAgentLLM
from app.infrastructure.llm import OllamaLLM


class SlowHeadersTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        await asyncio.sleep(0.2)
        raise AssertionError("The absolute deadline should cancel this request")


@pytest.mark.parametrize("adapter,error,call", [
    (
        OllamaLLM(
            "http://127.0.0.1:11434", "test", timeout_seconds=0.04,
            transport=SlowHeadersTransport(),
        ),
        LLMTimeout,
        lambda adapter: adapter.generate("system", "user", ["S1"]),
    ),
    (
        OllamaAgentLLM(
            "http://127.0.0.1:11434", "test", timeout_seconds=0.04,
            transport=SlowHeadersTransport(),
        ),
        AgentLLMTimeout,
        lambda adapter: adapter.generate("system", "user"),
    ),
])
def test_absolute_deadline_cancels_waiting_for_response_headers(adapter, error, call):
    started = perf_counter()
    with pytest.raises(error):
        call(adapter)
    assert perf_counter() - started < 0.5


def test_single_connect_retry_succeeds_without_retrying_generation():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("not connected", request=request)
        return httpx.Response(200, json={
            "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps({
                "answer": "Connected on the bounded retry."
            })},
        })

    result = OllamaAgentLLM(
        "http://127.0.0.1:11434", "test", connect_retries=1,
        transport=httpx.MockTransport(handler),
    ).generate("system", "user")
    assert result.answer == "Connected on the bounded retry."
    assert calls == 2


@pytest.mark.parametrize("failure,expected", [
    (httpx.ReadTimeout, AgentLLMTimeout),
    (httpx.ReadError, AgentLLMUnavailable),
])
def test_read_failures_are_never_retried(failure, expected):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise failure("read failed", request=request)

    adapter = OllamaAgentLLM(
        "http://127.0.0.1:11434", "test", connect_retries=2,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(expected):
        adapter.generate("system", "user")
    assert calls == 1
