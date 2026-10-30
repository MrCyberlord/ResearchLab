"""Native Ollama structured generation; no fallback answers or remote provider SDK."""
import json
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from app.domain.rag import (
    LLMInvalidResponse, LLMResult, LLMTimeout, LLMUnavailable, SourceReference, TokenUsage,
)
from app.infrastructure.config import Settings
from app.infrastructure.ollama_http import (
    BoundedOllamaHTTPClient, OllamaHTTPDeadlineExceeded, OllamaHTTPInvalidResponse,
    OllamaHTTPUnavailable,
)

GROUNDED_ANSWER_SCHEMA_VERSION = "grounded-answer-schema-v2"


class _Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_id: str = Field(min_length=1, max_length=80)
    quote: str = Field(min_length=1, max_length=1000)

    @field_validator("quote")
    @classmethod
    def nonblank_quote(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Supporting quote cannot be blank")
        return value


class _Answered(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["answered"]
    answer: str = Field(min_length=1, max_length=12000)
    evidence: list[_Evidence] = Field(min_length=1, max_length=20)

    @field_validator("answer")
    @classmethod
    def nonblank_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Answer cannot be blank")
        return value.strip()


class _InsufficientEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["insufficient_evidence"]


_StructuredAnswer = Annotated[_Answered | _InsufficientEvidence, Field(discriminator="status")]
_STRUCTURED_ANSWER_ADAPTER = TypeAdapter(_StructuredAnswer)


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
                 max_response_bytes: int = 65536, connect_retries: int = 1, *,
                 transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None):
        self.base_url = str(base_url).rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.context_tokens = context_tokens
        self.max_output_tokens = max_output_tokens
        self.max_response_bytes = max_response_bytes
        self.connect_retries = connect_retries
        self._transport = transport

    def generate(self, system_prompt: str, user_prompt: str, source_ids: list[str],
                 *, timeout_seconds: float | None = None) -> LLMResult:
        if (not source_ids or len(source_ids) > 20
                or any(not isinstance(item, str) or not item or len(item) > 80 for item in source_ids)
                or len(set(source_ids)) != len(source_ids)):
            raise LLMInvalidResponse("Generation requires valid context source labels.")
        schema = _STRUCTURED_ANSWER_ADAPTER.json_schema()
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
        timeout = self.timeout_seconds if timeout_seconds is None else min(
            self.timeout_seconds, timeout_seconds,
        )
        if timeout <= 0:
            raise LLMTimeout("Ollama generation timed out. Try a smaller context or a faster local model.")
        try:
            data = BoundedOllamaHTTPClient(
                self.base_url, self.max_response_bytes, self.connect_retries,
                transport=self._transport,
            ).post_chat(request_body, timeout)
        except OllamaHTTPDeadlineExceeded:
            raise LLMTimeout("Ollama generation timed out. Try a smaller context or a faster local model.") from None
        except OllamaHTTPUnavailable:
            raise LLMUnavailable("Ollama is unavailable. Check the configured local service and model.") from None
        except OllamaHTTPInvalidResponse:
            raise LLMInvalidResponse("Ollama returned an invalid or oversized response.") from None

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
            answer = _STRUCTURED_ANSWER_ADAPTER.validate_python(_strict_json(message["content"]))
            if isinstance(answer, _Answered) and any(
                item.source_id not in source_ids for item in answer.evidence
            ):
                raise ValueError("Unknown evidence source")
            prompt_tokens = _token_count(payload, "prompt_eval_count")
            completion_tokens = _token_count(payload, "eval_count")
        except (ValueError, TypeError, UnicodeError, ValidationError):
            raise LLMInvalidResponse("Ollama returned an invalid or incomplete structured answer.") from None

        usage = (TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
                 if prompt_tokens is not None or completion_tokens is not None else None)
        if isinstance(answer, _InsufficientEvidence):
            return LLMResult(answer="", evidence=[], insufficient_evidence=True, usage=usage)
        return LLMResult(
            answer=answer.answer,
            evidence=[SourceReference(item.source_id, item.quote) for item in answer.evidence],
            insufficient_evidence=False, usage=usage,
        )


def create_llm_provider(settings: Settings) -> OllamaLLM:
    return OllamaLLM(settings.llm_base_url, settings.llm_model, settings.llm_timeout_seconds,
                     settings.llm_context_tokens, settings.llm_max_output_tokens,
                     settings.llm_max_response_bytes, settings.llm_connect_retries)
