from contextlib import contextmanager
from uuid import uuid4
import math

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.dependencies import get_ingestion_service
from app.application.ingestion import IngestionService, split_pages
from app.domain.documents import DuplicateDocument, EmbeddingFailure, InvalidPDF, PDFLimitExceeded, PageText, IncompatibleEmbeddings
from app.infrastructure.config import Settings
from app.infrastructure.embeddings import LocalHashEmbeddings, create_embedding_provider
from app.infrastructure.pdf import IsolatedPDFExtractor
from app.main import create_app


class MemoryRepository:
    def __init__(self):
        self.documents = []
        self.chunks = []

    @contextmanager
    def transaction(self):
        yield

    def check_profile(self, profile):
        if getattr(self, "profile", profile) != profile:
            raise IncompatibleEmbeddings()

    def bind_profile(self, profile):
        self.check_profile(profile)
        self.profile = profile

    def find_by_hash(self, digest):
        return next((document for document in self.documents if document.content_hash == digest), None)

    def add(self, document, chunks, vectors):
        self.documents.append(document)
        self.chunks.extend(chunks)


class StaticExtractor:
    def extract(self, data):
        return [PageText(1, "A" * 120), PageText(2, "B" * 40)]


def service(repository=None, extractor=None, embeddings=None):
    return IngestionService(repository or MemoryRepository(), extractor or StaticExtractor(),
                            embeddings or LocalHashEmbeddings(), 100, 20)


def test_chunks_keep_page_id_order_and_overlap():
    document_id = uuid4()
    chunks = split_pages([PageText(1, "abcdefghij"), PageText(2, "klmnop")], document_id, 6, 2)
    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij", "klmnop"]
    assert [chunk.page_number for chunk in chunks] == [1, 1, 2]
    assert [chunk.chunk_order for chunk in chunks] == [0, 1, 2]
    assert all(chunk.document_id == document_id for chunk in chunks)
    assert split_pages([PageText(1, " \x00 ")], document_id, 6, 2) == []
    with pytest.raises(PDFLimitExceeded):
        split_pages([PageText(1, "abcdefghij")], document_id, 6, 2, max_chunks=1)


@pytest.mark.parametrize("config", [{"chunk_overlap": 1000}, {"max_upload_bytes": 0},
                                      {"embedding_dimension": 128}, {"embedding_model": "unknown"}])
def test_invalid_ingestion_config(config):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **config)


def test_local_vectors_are_stable_and_normalized():
    settings = Settings(_env_file=None, embedding_provider="local", embedding_model="local-hash-v1")
    provider = create_embedding_provider(settings)
    assert provider.dimension == settings.embedding_dimension
    assert provider.model == settings.embedding_model
    vectors = provider.embed(["Research LAB", "research lab", "something else"])
    assert vectors[0] == vectors[1]
    assert vectors[0] != vectors[2]
    assert len(vectors[0]) == 384
    assert math.isclose(sum(x*x for x in vectors[0]), 1.0)


def test_service_metadata_duplicate_and_safe_filename():
    repository = MemoryRepository()
    ingestion = service(repository)
    document = ingestion.ingest(b"pdf bytes", "../../report.pdf")
    assert document.filename == "report.pdf"
    assert document.page_count == 2 and document.chunk_count == 3
    assert document.embedding_model == "local-hash-v1"
    with pytest.raises(DuplicateDocument) as error:
        ingestion.ingest(b"pdf bytes", "another-name.pdf")
    assert error.value.document_id == document.id
    assert len(repository.documents) == 1


@pytest.mark.parametrize("bad_vectors", [[], [[0.0] * 384] * 3, [[1.0] * 10] * 3, [[float('nan')] * 384] * 3])
def test_embedding_failure_never_writes(bad_vectors):
    class BrokenEmbeddings(LocalHashEmbeddings):
        def embed(self, texts): return bad_vectors
    repository = MemoryRepository()
    with pytest.raises(EmbeddingFailure):
        service(repository, embeddings=BrokenEmbeddings()).ingest(b"data", "report.pdf")
    assert repository.documents == [] and repository.chunks == []


def test_extraction_failure_never_writes():
    class BrokenExtractor:
        def extract(self, data): raise InvalidPDF("Cannot parse PDF")
    repository = MemoryRepository()
    with pytest.raises(InvalidPDF):
        service(repository, extractor=BrokenExtractor()).ingest(b"bad", "report.pdf")
    assert not repository.documents


def test_isolated_pdf_extraction(pdf_bytes):
    extractor = IsolatedPDFExtractor(10, 10000, 15, 256)
    pages = extractor.extract(pdf_bytes(["Page one", "Page two"]))
    assert [(p.number, p.text.strip()) for p in pages] == [(1, "Page one"), (2, "Page two")]
    with pytest.raises(InvalidPDF, match="Encrypted"):
        extractor.extract(pdf_bytes(encrypted=True))
    with pytest.raises(InvalidPDF):
        extractor.extract(b"%PDF-1.7 corrupt")
    with pytest.raises(InvalidPDF, match="header"):
        extractor.extract(b"not pdf")


def test_pdf_limits(pdf_bytes):
    data = pdf_bytes(["text", "text"])
    with pytest.raises(PDFLimitExceeded, match="page"):
        IsolatedPDFExtractor(1, 10000, 15, 256).extract(data)
    with pytest.raises(PDFLimitExceeded, match="text"):
        IsolatedPDFExtractor(10, 1, 15, 256).extract(data)
    with pytest.raises(PDFLimitExceeded, match="time"):
        IsolatedPDFExtractor(10, 10000, 0.000001, 256).extract(data)
    with pytest.raises(PDFLimitExceeded, match="memory"):
        IsolatedPDFExtractor(10, 10000, 15, 1).extract(data)


def test_blank_pdf_rejected(pdf_bytes):
    with pytest.raises(InvalidPDF, match="no extractable text"):
        service(extractor=IsolatedPDFExtractor(10, 10000, 15, 256)).ingest(pdf_bytes([""]), "blank.pdf")


@pytest.fixture
def api():
    app = create_app(Settings(_env_file=None, database_url=None, max_upload_bytes=1024))
    app.dependency_overrides[get_ingestion_service] = lambda: service()
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("filename,mime,data,status,code", [
    ("test.txt", "application/pdf", b"pdf", 415, "unsupported_type"),
    ("test.pdf", "text/plain", b"pdf", 415, "unsupported_type"),
    ("test.pdf", "application/pdf", b"", 422, "empty_file"),
    ("test.pdf", "application/pdf", b"x" * 1025, 413, "file_too_large"),
])
def test_upload_validation(api, filename, mime, data, status, code):
    response = api.post("/api/documents", files={"file": (filename, data, mime)})
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code


def test_request_body_limit_without_content_length(api):
    response = api.post("/api/documents", content=iter([b"x" * 40000, b"x" * 40000]),
                        headers={"Content-Type": "multipart/form-data; boundary=x"})
    assert response.status_code == 413


def test_request_body_limit_with_content_length(api):
    response = api.post("/api/documents", files={"file": ("large.pdf", b"x" * 70000, "application/pdf")},
                        headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 413
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_missing_upload(api):
    assert api.post("/api/documents").status_code == 422


def test_unconfigured_database():
    with TestClient(create_app(Settings(_env_file=None, database_url=None))) as client:
        assert client.post("/api/documents", files={"file": ("test.pdf", b"pdf", "application/pdf")}).status_code == 503


@pytest.mark.parametrize("failure,status,code", [
    (DuplicateDocument(), 409, "duplicate_document"),
    (IncompatibleEmbeddings(), 409, "embedding_profile_mismatch"),
    (InvalidPDF("Malformed PDF"), 422, "invalid_pdf"),
    (PDFLimitExceeded("Page limit"), 422, "pdf_limit_exceeded"),
    (EmbeddingFailure("Embedding unavailable"), 503, "embedding_failed"),
])
def test_ingestion_errors_are_meaningful(api, failure, status, code):
    class FailingService:
        def ingest(self, data, filename): raise failure
    api.app.dependency_overrides[get_ingestion_service] = lambda: FailingService()
    response = api.post("/api/documents", files={"file": ("test.pdf", b"pdf", "application/pdf")})
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code


def test_storage_error_does_not_leak_database_details(api):
    from sqlalchemy.exc import SQLAlchemyError
    class FailingService:
        def ingest(self, data, filename): raise SQLAlchemyError("sensitive database details")
    api.app.dependency_overrides[get_ingestion_service] = lambda: FailingService()
    response = api.post("/api/documents", files={"file": ("test.pdf", b"pdf", "application/pdf")})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "storage_unavailable"
    assert "sensitive" not in response.text


def test_incompatible_profile_rejected_before_extraction():
    repository = MemoryRepository()
    repository.profile = "another-model"
    with pytest.raises(IncompatibleEmbeddings):
        service(repository).ingest(b"data", "document.pdf")
    assert repository.documents == []
