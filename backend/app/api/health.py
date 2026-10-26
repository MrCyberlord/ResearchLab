from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal

router = APIRouter(prefix="/api", tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "researchlab-api"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """API liveness only; does not claim database or provider readiness."""
    return HealthResponse()
