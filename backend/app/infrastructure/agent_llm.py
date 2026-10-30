"""Ollama adapter for ungrounded Agentic responses; separate from RAG citations."""
import json
from time import perf_counter

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.domain.agent import (
    AgentGeneration, AgentLLMInvalidResponse, AgentLLMTimeout,
    AgentLLMUnavailable, AgentTokenUsage,
)
from app.infrastructure.config import Settings


def _strict_json(value: str):
    def unique_keys(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = item
        return result

    def invalid_constant(value):
        raise ValueError("Invalid JSON numeric constant")

    return json.loads(value, object_pairs_hook=unique_keys, parse_constant=invalid_constant)


def _token_count(payload: dict, key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError("Invalid token count")
    return value


class _AgentAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str = Field(min_length=1, max_length=12000)

    @field_validator("answer")
    @classmethod
    def nonblank_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Answer cannot be blank")
        return value.strip()


class OllamaAgentLLM:
    provider = "ollama"

    def __init__(self, base_url: str, model: str, timeout_seconds: float = 120,
                 context_tokens: int = 8192, max_output_tokens: int = 384,
                 max_response_bytes: int = 65536, *, transport: httpx.BaseTransport | None = None):
        self.base_url = str(base_url).rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.context_tokens = context_tokens
        self.max_output_tokens = max_output_tokens
        self.max_response_bytes = max_response_bytes
        self._transport = transport

    def generate(self, system_prompt: str, user_prompt: str) -> AgentGeneration:
        request_body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
            "stream": False,
            "format": _AgentAnswer.model_json_schema(),
            "options": {"temperature": 0, "num_ctx": self.context_tokens,
                        "num_predict": self.max_output_tokens},
        }
        started = perf_counter()
        data = bytearray()
        try:
            with httpx.Client(timeout=httpx.Timeout(self.timeout_seconds, connect=min(5, self.timeout_seconds)),
                              trust_env=False, follow_redirects=False, transport=self._transport) as client:
                with client.stream("POST", self.base_url + "/api/chat", json=request_body,
                                   headers={"Accept": "application/json", "Accept-Encoding": "identity"}) as response:
                    if response.status_code != 200:
                        raise AgentLLMUnavailable("Ollama could not generate a response. Check the service and configured model.")
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise AgentLLMInvalidResponse("Ollama returned an unsupported response encoding.")
                    length = response.headers.get("content-length")
                    if length is not None:
                        try:
                            if int(length) < 0 or int(length) > self.max_response_bytes:
                                raise ValueError("Response size limit exceeded")
                        except ValueError:
                            raise AgentLLMInvalidResponse("Ollama returned an invalid or oversized response.") from None
                    for chunk in response.iter_bytes(chunk_size=16384):
                        if perf_counter() - started > self.timeout_seconds:
                            raise AgentLLMTimeout("Ollama generation timed out.")
                        data.extend(chunk)
                        if len(data) > self.max_response_bytes:
                            raise AgentLLMInvalidResponse("Ollama returned an oversized response.")
        except httpx.TimeoutException:
            raise AgentLLMTimeout("Ollama generation timed out.") from None
        except httpx.HTTPError:
            raise AgentLLMUnavailable("Ollama is unavailable. Check the configured local service and model.") from None

        try:
            payload = _strict_json(data.decode("utf-8"))
            if not isinstance(payload, dict) or payload.get("done") is not True:
                raise ValueError("Incomplete generation")
            if payload.get("done_reason") not in (None, "stop"):
                raise ValueError("Truncated generation")
            message = payload.get("message")
            if (not isinstance(message, dict) or message.get("role") != "assistant"
                    or not isinstance(message.get("content"), str)):
                raise ValueError("Invalid assistant message")
            answer = _AgentAnswer.model_validate(_strict_json(message["content"]))
            prompt_tokens = _token_count(payload, "prompt_eval_count")
            completion_tokens = _token_count(payload, "eval_count")
        except (ValueError, TypeError, UnicodeError, ValidationError):
            raise AgentLLMInvalidResponse("Ollama returned an invalid or incomplete structured response.") from None
        usage = (AgentTokenUsage(prompt_tokens, completion_tokens)
                 if prompt_tokens is not None or completion_tokens is not None else None)
        return AgentGeneration(answer.answer, usage)


def create_agent_generation_provider(settings: Settings) -> OllamaAgentLLM:
    return OllamaAgentLLM(settings.llm_base_url, settings.llm_model,
                          settings.agent_execution_timeout_seconds,
                          settings.llm_context_tokens, settings.agent_max_output_tokens,
                          settings.agent_max_response_bytes)
