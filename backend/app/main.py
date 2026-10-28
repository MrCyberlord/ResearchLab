from contextlib import asynccontextmanager
import logging
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router
from app.api.documents import router as document_router
from app.api.upload_limits import UploadBodyLimit
from app.api.rag import router as rag_router
from app.infrastructure.config import Settings, get_settings
from app.infrastructure.database import create_session_factory
from app.infrastructure.logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else get_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger("researchlab")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = None
        app.state.session_factory = None
        if settings.database_url:
            engine, app.state.session_factory = create_session_factory(settings)
        logger.info("application_started")
        try:
            yield
        finally:
            if engine is not None:
                engine.dispose()
            logger.info("application_stopped")

    app = FastAPI(title="ResearchLab API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(UploadBodyLimit, max_bytes=settings.max_upload_bytes + 64 * 1024)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_failed", extra={"method": request.method})
            raise
        logger.info("request_completed", extra={"method": request.method,
                    "status_code": response.status_code,
                    "duration_ms": round((perf_counter() - started) * 1000, 2)})
        return response

    app.include_router(router)
    app.include_router(document_router)
    app.include_router(rag_router)
    return app


app = create_app()
