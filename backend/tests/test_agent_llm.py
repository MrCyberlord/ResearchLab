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


def test_agent_ollama_tool_decision_is_model_generated_and_structured():
    captured = []
    responses = iter([
        payload(message={"role": "assistant", "content": json.dumps({
            "scope": "document_research",
        })}),
        payload(message={"role": "assistant", "content": json.dumps({
            "action": "search_documents", "reason_code": "evidence_needed",
            "query": "kettle warranty duration", "top_k": 4,
        })}),
    ])
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=next(responses))
    result = provider(handler).decide(
        "What do my documents say?", [{"searches": [], "evidence": []}], 3,
    )
    assert result.action == "search_documents"
    assert result.scope == "document_research"
    assert result.reason_code == "evidence_needed"
    assert result.query == "kettle warranty duration" and result.top_k == 4
    assert len(captured) == 2
    scope_user = json.loads(captured[0]["messages"][1]["content"])
    assert scope_user == {"question": "What do my documents say?"}
    assert set(captured[0]["format"]["properties"]) == {"scope"}
    user = json.loads(captured[1]["messages"][1]["content"])
    assert user["question"] == "What do my documents say?"
    assert user["document_search_calls_remaining"] == 3
    assert "search_documents" in captured[1]["messages"][0]["content"]
    assert captured[1]["format"]["discriminator"]["propertyName"] == "action"


def test_general_scope_returns_direct_decision_without_tool_schema_call():
    captured = []
    body = payload(message={"role": "assistant", "content": json.dumps({
        "scope": "general_knowledge",
    })})
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=body)
    decision = provider(handler).decide(
        "what is llm", [{"searches": [], "evidence": []}], 3,
    )
    assert decision.action == "answer_direct" and decision.scope == "general_knowledge"
    assert decision.reason_code == "general_question"
    assert decision.query is None and decision.top_k is None
    assert len(captured) == 1


@pytest.mark.parametrize("decision", [
    {"action": "search_documents", "query": None, "top_k": 2},
    {"action": "search_documents", "query": "query", "top_k": 21},
    {"action": "answer_direct"},
    {"action": "unknown"},
    {"action": "answer_grounded", "query": "E1", "top_k": 1},
    {"action": "answer_grounded", "reason_code": "refine_search"},
    {"action": "insufficient_evidence", "reason_code": "evidence_supports_answer"},
])
def test_invalid_structured_tool_decisions_are_rejected(decision):
    responses = iter([
        payload(message={"role": "assistant", "content": json.dumps({
            "scope": "document_research",
        })}),
        payload(message={"role": "assistant", "content": json.dumps(decision)}),
    ])
    with pytest.raises(AgentLLMInvalidResponse):
        provider(lambda request: httpx.Response(200, json=next(responses))).decide(
            "question", [], 2,
        )


def test_document_terminal_action_has_no_permissive_tool_argument_fallback():
    body = payload(message={"role": "assistant", "content": json.dumps({
        "action": "answer_grounded", "reason_code": "evidence_supports_answer",
    })})
    decision = provider(lambda request: httpx.Response(200, json=body)).decide(
        "question", [{"searches": [{"query": "q"}], "evidence": [{"text": "e"}]}], 2,
    )
    assert decision.action == "answer_grounded"
    assert decision.reason_code == "evidence_supports_answer"
    assert decision.query is None and decision.top_k is None


def test_paraphrased_document_evidence_has_explicit_grounded_decision_contract():
    captured = {}
    body = payload(message={"role": "assistant", "content": json.dumps({
        "action": "answer_grounded", "reason_code": "evidence_supports_answer",
    })})

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=body)

    decision = provider(handler).decide(
        "What happens when stored electricity falls below its threshold?",
        [{"searches": [{"result_count": 2, "outcome": "results"}], "evidence": [{
            "evidence_id": "E1",
            "text": "If battery charge drops below 28 percent, the workshop disconnects.",
        }]}],
        2,
    )
    assert decision.action == "answer_grounded"
    assert decision.reason_code == "evidence_supports_answer"
    prompt = captured["body"]["messages"][0]["content"]
    assert "paraphrases or uses synonyms" in prompt
    assert "Do not require an exact word match" in prompt
    schema = json.dumps(captured["body"]["format"], sort_keys=True)
    assert "evidence_supports_answer" in schema
    assert "evidence_does_not_support_answer" in schema


def test_search_reason_schema_changes_after_document_observation():
    initial_responses = iter([
        payload(message={"role": "assistant", "content": json.dumps({
            "scope": "document_research",
        })}),
        payload(message={"role": "assistant", "content": json.dumps({
            "action": "search_documents", "reason_code": "refine_search",
            "query": "refine too early", "top_k": 2,
        })}),
    ])
    with pytest.raises(AgentLLMInvalidResponse):
        provider(lambda request: httpx.Response(
            200, json=next(initial_responses),
        )).decide("What does my document say?", [], 3)

    observed = payload(message={"role": "assistant", "content": json.dumps({
        "action": "search_documents", "reason_code": "evidence_needed",
        "query": "initial reason too late", "top_k": 2,
    })})
    with pytest.raises(AgentLLMInvalidResponse):
        provider(lambda request: httpx.Response(200, json=observed)).decide(
            "What does my document say?",
            [{"searches": [{"outcome": "results"}], "evidence": [{"text": "fact"}]}],
            2,
        )


def test_invalid_scope_decision_is_rejected_before_document_action():
    body = payload(message={"role": "assistant", "content": json.dumps({
        "scope": "general_knowledge", "action": "answer_direct",
    })})
    with pytest.raises(AgentLLMInvalidResponse):
        provider(lambda request: httpx.Response(200, json=body)).decide("question", [], 2)


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
