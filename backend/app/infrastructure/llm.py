"""Native Ollama structured generation; no fallback answers or remote provider SDK."""
import json
from time import perf_counter

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.domain.rag import (
    LLMInvalidResponse, LLMResult, LLMTimeout, LLMUnavailable, SourceReference, TokenUsage,
)
from app.infrastructure.config import Settings


class _Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_id: str = Field(min_length=1, max_length=80)
    quote: str = Field(min_length=1, max_length=1000)


class _StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str = Field(max_length=12000)
    evidence: list[_Evidence] = Field(max_length=20)
    insufficient_evidence: bool

    @model_validator(mode="after")
    def require_supported_answer(self):
        if any(not item.quote.strip() for item in self.evidence):
            raise ValueError("Supporting quotes cannot be blank")
        if not self.insufficient_evidence and (not self.answer.strip() or not self.evidence):
            raise ValueError("An answered question requires an answer and supporting evidence")
        return self


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


class OllamaLLM:
    provider = "ollama"

    def __init__(self, base_url: str, model: str, timeout_seconds: float = 120,
                 context_tokens: int = 8192, max_output_tokens: int = 512,
                 max_response_bytes: int = 65536, *, transport: httpx.BaseTransport | None = None):
        self.base_url = str(base_url).rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.context_tokens = context_tokens
        self.max_output_tokens = max_output_tokens
        self.max_response_bytes = max_response_bytes
        self._transport = transport

    def generate(self, system_prompt: str, user_prompt: str, source_ids: list[str]) -> LLMResult:
        if (not source_ids or len(source_ids) > 20
                or any(not isinstance(item, str) or not item or len(item) > 80 for item in source_ids)
                or len(set(source_ids)) != len(source_ids)):
            raise LLMInvalidResponse("Generation requires valid context source labels.")
        schema = _StructuredAnswer.model_json_schema()
        schema["$defs"]["_Evidence"]["properties"]["source_id"]["enum"] = source_ids
        request_body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
            "stream": False,
            "format": schema,
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
                        raise LLMUnavailable("Ollama could not generate an answer. Check the service and configured model.")
                    # Avoid decoding an unexpectedly compressed response before its size can be bounded.
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise LLMInvalidResponse("Ollama returned an unsupported response encoding.")
                    length = response.headers.get("content-length")
                    if length is not None:
                        try:
                            if int(length) < 0 or int(length) > self.max_response_bytes:
                                raise ValueError("Response size limit exceeded")
                        except ValueError:
                            raise LLMInvalidResponse("Ollama returned an invalid or oversized response.") from None
                    for chunk in response.iter_bytes(chunk_size=16384):
                        if perf_counter() - started > self.timeout_seconds:
                            raise LLMTimeout("Ollama generation timed out. Try a smaller context or a faster local model.")
                        data.extend(chunk)
                        if len(data) > self.max_response_bytes:
                            raise LLMInvalidResponse("Ollama returned an oversized response.")
        except httpx.TimeoutException:
            raise LLMTimeout("Ollama generation timed out. Try a smaller context or a faster local model.") from None
        except httpx.HTTPError:
            raise LLMUnavailable("Ollama is unavailable. Check the configured local service and model.") from None

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
            answer = _StructuredAnswer.model_validate(_strict_json(message["content"]))
            if any(item.source_id not in source_ids for item in answer.evidence):
                raise ValueError("Unknown evidence source")
            prompt_tokens = _token_count(payload, "prompt_eval_count")
            completion_tokens = _token_count(payload, "eval_count")
        except (ValueError, TypeError, UnicodeError, ValidationError):
            raise LLMInvalidResponse("Ollama returned an invalid or incomplete structured answer.") from None

        usage = (TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
                 if prompt_tokens is not None or completion_tokens is not None else None)
        return LLMResult(answer=answer.answer.strip(),
                         evidence=[SourceReference(item.source_id, item.quote) for item in answer.evidence],
                         insufficient_evidence=answer.insufficient_evidence, usage=usage)


def create_llm_provider(settings: Settings) -> OllamaLLM:
    return OllamaLLM(settings.llm_base_url, settings.llm_model, settings.llm_timeout_seconds,
                     settings.llm_context_tokens, settings.llm_max_output_tokens,
                     settings.llm_max_response_bytes)
