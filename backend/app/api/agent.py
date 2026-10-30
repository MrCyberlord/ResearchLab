"""HTTP boundary for the bounded, ungrounded Milestone 3A agent workflow."""
from contextlib import contextmanager
from dataclasses import asdict
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.api.dependencies import get_agent_service
from app.application.agent import AgentService
from app.domain.agent import (
    AgentExecutionLimit, AgentLLMInvalidResponse, AgentLLMTimeout,
    AgentLLMUnavailable, AgentQuestionInvalid,
)

router = APIRouter(prefix="/api/agent", tags=["agentic"])
AgentQuestion = Annotated[str, StringConstraints(
    strict=True, strip_whitespace=True, min_length=1, max_length=10000,
)]
DISCLAIMER = "This response is not document-grounded. Document research tools and citations are not available in Milestone 3A."


class AgentQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: AgentQuestion
    include_metadata: bool = Field(default=True, strict=True)


class AgentUsageResponse(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


class AgentNodeTraceResponse(BaseModel):
    node: str
    duration_ms: float = Field(ge=0)
    outcome: str


class AgentMetadataResponse(BaseModel):
    route: Literal["direct", "document_tools_unavailable"]
    duration_ms: float = Field(ge=0)
    steps: int = Field(ge=1)
    node_trace: list[AgentNodeTraceResponse]
    usage: AgentUsageResponse | None


class AgentQuestionResponse(BaseModel):
    answer: str
    provider: str
    model: str
    document_tools_available: Literal[False]
    disclaimer: str
    metadata: AgentMetadataResponse | None


class AgentConfigResponse(BaseModel):
    provider: str
    model: str
    max_question_chars: int
    graph_step_limit: int
    execution_timeout_seconds: float
    document_tools_available: Literal[False] = False


@contextmanager
def agent_errors():
    try:
        yield
    except AgentQuestionInvalid as exc:
        raise HTTPException(422, detail={"code": "invalid_question", "message": str(exc)}) from exc
    except AgentLLMTimeout as exc:
        raise HTTPException(504, detail={"code": "agent_llm_timeout", "message": "The configured local model timed out."}) from exc
    except AgentLLMUnavailable as exc:
        raise HTTPException(503, detail={"code": "agent_llm_unavailable", "message": "The configured local LLM or model is unavailable."}) from exc
    except AgentLLMInvalidResponse as exc:
        raise HTTPException(502, detail={"code": "invalid_agent_response", "message": "The local model returned an invalid response."}) from exc
    except AgentExecutionLimit as exc:
        raise HTTPException(500, detail={"code": "agent_execution_limit", "message": "The bounded agent execution limit was reached."}) from exc


@router.get("/config", response_model=AgentConfigResponse)
def agent_config(request: Request):
    settings = request.app.state.settings
    return AgentConfigResponse(provider=settings.llm_provider, model=settings.llm_model,
                               max_question_chars=settings.agent_max_question_chars,
                               graph_step_limit=settings.agent_graph_step_limit,
                               execution_timeout_seconds=settings.agent_execution_timeout_seconds)


@router.post("/questions", response_model=AgentQuestionResponse)
def ask_agent(body: AgentQuestionRequest, service: AgentService = Depends(get_agent_service)):
    with agent_errors():
        result = service.run(body.question, body.include_metadata)
        return {**asdict(result), "disclaimer": DISCLAIMER}
