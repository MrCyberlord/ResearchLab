from typing import Annotated
from uuid import UUID
import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import get_ingestion_service
from app.application.ingestion import IngestionService
from app.domain.documents import DuplicateDocument, EmbeddingFailure, InvalidPDF, PDFLimitExceeded, IncompatibleEmbeddings

router = APIRouter(prefix="/api/documents", tags=["documents"])
logger = logging.getLogger("researchlab")


class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    page_count: int
    chunk_count: int
    embedding_model: str
    embedding_dimension: int
    embedding_revision: str
    embedding_profile: str


def error(status: int, code: str, message: str, **extra):
    return HTTPException(status_code=status, detail={"code": code, "message": message, **extra})


@router.post("", status_code=201, response_model=DocumentResponse)
def upload_document(request: Request, file: Annotated[UploadFile, File()],
                    service: Annotated[IngestionService, Depends(get_ingestion_service)]):
    try:
        if file.content_type != "application/pdf" or not (file.filename or "").lower().endswith(".pdf"):
            raise error(415, "unsupported_type", "Upload a .pdf file with application/pdf content type.")
        limit = request.app.state.settings.max_upload_bytes
        data = file.file.read(limit + 1)
        if len(data) > limit:
            raise error(413, "file_too_large", f"PDF must be at most {limit} bytes.")
        if not data:
            raise error(422, "empty_file", "The uploaded PDF is empty.")
        document = service.ingest(data, file.filename or "document.pdf")
        return DocumentResponse(**{field: getattr(document, field) for field in DocumentResponse.model_fields})
    except IncompatibleEmbeddings as exc:
        raise error(409, "embedding_profile_mismatch", str(exc)) from exc
    except DuplicateDocument as exc:
        raise error(409, "duplicate_document", str(exc), document_id=str(exc.document_id) if exc.document_id else None) from exc
    except PDFLimitExceeded as exc:
        raise error(422, "pdf_limit_exceeded", str(exc)) from exc
    except InvalidPDF as exc:
        raise error(422, "invalid_pdf", str(exc)) from exc
    except EmbeddingFailure as exc:
        raise error(503, "embedding_failed", str(exc)) from exc
    except SQLAlchemyError as exc:
        logger.error("ingestion_storage_failed")
        raise error(503, "storage_unavailable", "Document could not be saved. Retry later.") from exc
    finally:
        file.file.close()
