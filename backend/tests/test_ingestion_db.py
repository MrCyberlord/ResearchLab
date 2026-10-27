"""Opt-in tests against migrated PostgreSQL. Every test rolls back its own writes."""
import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.api.dependencies import get_session
from app.application.ingestion import IngestionService
from app.domain.documents import Chunk, Document, DuplicateDocument, EmbeddingFailure, InvalidPDF
from app.infrastructure.config import Settings
from app.infrastructure.database import Base, create_session_factory
from app.infrastructure.document_models import ChunkRow, DocumentRow
from app.infrastructure.document_repository import SQLAlchemyDocumentRepository
from app.infrastructure.embeddings import LocalHashEmbeddings
from app.infrastructure.pdf import IsolatedPDFExtractor
from app.main import create_app

pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.getenv("RUN_DB_TESTS") != "1", reason="Set RUN_DB_TESTS=1 to use migrated local PostgreSQL")]


@pytest.fixture
def db():
    engine, _ = create_session_factory(Settings())
    with engine.connect() as connection:
        transaction = connection.begin()
        session = None
        try:
            # Tests must not depend on, lock, or rewrite a user's public corpus.
            # All DDL and data in this private schema disappear with outer rollback.
            schema = "test_ingestion_" + uuid4().hex
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            connection.execute(text(f'SET LOCAL search_path TO "{schema}", public'))
            # checkfirst=True could find public tables through search_path and skip
            # creating their private counterparts. Public migration drift is checked
            # separately with Alembic against the real application schema.
            Base.metadata.create_all(connection, checkfirst=False)
            connection.execute(text("CREATE TABLE alembic_version (version_num varchar(32) PRIMARY KEY)"))
            connection.execute(text("INSERT INTO alembic_version SELECT version_num FROM public.alembic_version"))
            session = Session(bind=connection, join_transaction_mode="create_savepoint")
            yield session
        finally:
            if session is not None:
                session.close()
            transaction.rollback()
    engine.dispose()


def test_database_fixture_isolates_public_corpus(db):
    assert db.scalar(text("SELECT current_schema()" )).startswith("test_ingestion_")
    for table in ("documents", "chunks", "embedding_corpus"):
        assert db.scalar(text("SELECT to_regclass(:local)::oid <> to_regclass(:public)::oid"),
                         {"local": table, "public": "public." + table})


def counts(db):
    result = (db.scalar(select(func.count()).select_from(DocumentRow)),
              db.scalar(select(func.count()).select_from(ChunkRow)))
    db.commit()
    return result


def ingestion(db, embeddings=None, repository=None):
    return IngestionService(repository or SQLAlchemyDocumentRepository(db),
                            IsolatedPDFExtractor(10, 10000, 15, 256),
                            embeddings or LocalHashEmbeddings(), 100, 20)


def test_schema_and_vector(db):
    assert db.scalar(text("SELECT version_num FROM alembic_version")) == "0003_embedding_profiles"
    assert db.scalar(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))
    assert db.scalar(text("SELECT '[1,2]'::vector <-> '[2,2]'::vector")) == 1


def test_upload_persists_metadata_vectors_and_duplicate(db, pdf_bytes):
    app = create_app(Settings(chunk_size=100, chunk_overlap=20, embedding_provider="local", embedding_model="local-hash-v1"))
    app.dependency_overrides[get_session] = lambda: db
    data = pdf_bytes(["A" * 120, "B" * 40])
    with TestClient(app) as client:
        response = client.post("/api/documents", files={"file": ("../report.pdf", data, "application/pdf")})
        assert response.status_code == 201, response.text
        result = response.json()
        assert result["filename"] == "report.pdf"
        assert result["page_count"] == 2 and result["chunk_count"] == 3
        rows = list(db.scalars(select(ChunkRow).where(ChunkRow.document_id == result["id"]).order_by(ChunkRow.chunk_order)))
        assert [r.page_number for r in rows] == [1, 1, 2]
        assert [r.chunk_order for r in rows] == [0, 1, 2]
        assert rows[0].text[-20:] == rows[1].text[:20]
        assert all(len(r.embedding) == 384 for r in rows)
        # End the inspection transaction before reusing the request session.
        db.commit()
        duplicate = client.post("/api/documents", files={"file": ("renamed.pdf", data, "application/pdf")})
        assert duplicate.status_code == 409
        assert duplicate.json()["detail"]["document_id"] == result["id"]


def test_database_rolls_back_partial_write(db, pdf_bytes):
    baseline = counts(db)
    class FailingRepository(SQLAlchemyDocumentRepository):
        def add(self, document, chunks, vectors):
            super().add(document, chunks, vectors)
            raise RuntimeError("Injected failure after flush")
    with pytest.raises(RuntimeError):
        ingestion(db, repository=FailingRepository(db)).ingest(pdf_bytes(), "rollback.pdf")
    assert counts(db) == baseline


def test_constraint_rolls_back_parent_on_bad_chunk(db, pdf_bytes):
    baseline = counts(db)
    class InvalidChunkRepository(SQLAlchemyDocumentRepository):
        def add(self, document, chunks, vectors):
            bad = Chunk(document.id, 0, 0, "invalid page")
            super().add(document, [bad], [vectors[0]])
    with pytest.raises(IntegrityError):
        ingestion(db, repository=InvalidChunkRepository(db)).ingest(pdf_bytes(), "bad.pdf")
    assert counts(db) == baseline


def test_unique_hash_race_maps_to_duplicate(db, pdf_bytes):
    baseline = counts(db)
    document = ingestion(db).ingest(pdf_bytes(), "original.pdf")
    duplicate = Document(uuid4(), "copy.pdf", document.content_hash, document.byte_size, 1, 1,
                         "local-hash-v1", 384, 100, 20)
    repository = SQLAlchemyDocumentRepository(db)
    with pytest.raises(DuplicateDocument):
        with repository.transaction():
            repository.add(duplicate, [Chunk(duplicate.id, 1, 0, "text")], [[1.0] * 384])
    assert counts(db)[0] == baseline[0] + 1


def test_extraction_and_embedding_failures_leave_no_rows(db, pdf_bytes):
    baseline = counts(db)
    with pytest.raises(InvalidPDF):
        ingestion(db).ingest(b"%PDF-1.7 malformed", "broken.pdf")
    class BrokenEmbeddings(LocalHashEmbeddings):
        def embed(self, texts): raise RuntimeError("provider unavailable")
    with pytest.raises(EmbeddingFailure):
        ingestion(db, embeddings=BrokenEmbeddings()).ingest(pdf_bytes(), "broken.pdf")
    assert counts(db) == baseline


def test_database_guards_against_mixed_profiles(db, pdf_bytes):
    from app.domain.documents import IncompatibleEmbeddings
    from app.infrastructure.embeddings import MiniLMEmbeddings
    from app.domain.embeddings import MINILM_MODEL, MINILM_REVISION
    ingestion(db).ingest(pdf_bytes(), "lexical.pdf")
    provider = MiniLMEmbeddings(MINILM_MODEL, MINILM_REVISION, "unused", True, 16, 2)
    with pytest.raises(IncompatibleEmbeddings):
        ingestion(db, embeddings=provider).ingest(pdf_bytes(), "semantic.pdf")
    assert counts(db)[0] >= 1


def test_foreign_key_prevents_bypassing_profile_guard(db, pdf_bytes):
    from dataclasses import replace
    document = ingestion(db).ingest(pdf_bytes(), "lexical.pdf")
    repository = SQLAlchemyDocumentRepository(db)
    incompatible = replace(document, id=uuid4(), content_hash="f" * 64, embedding_profile="a" * 64)
    with pytest.raises(IntegrityError):
        with repository.transaction():
            repository.add(incompatible, [Chunk(incompatible.id, 1, 0, "text")], [[1.0] * 384])


def test_reembedding_preserves_ids_text_and_is_atomic(db, pdf_bytes):
    from app.infrastructure.reembed import reembed_corpus
    from app.infrastructure.embedding_corpus import EmbeddingCorpusRow
    document = ingestion(db).ingest(pdf_bytes(), "lexical.pdf")
    ingestion(db).ingest(pdf_bytes(["A second document for atomic conversion"]), "second.pdf")
    old_chunks = [(r.id, r.text, r.page_number, r.chunk_order) for r in db.scalars(select(ChunkRow).order_by(ChunkRow.id))]
    db.commit()
    class NewProvider(LocalHashEmbeddings):
        model = "test-model"
        revision = "test-revision"
        profile = "b" * 64
    assert reembed_corpus(db, NewProvider()) >= 1
    stored = db.get(DocumentRow, document.id)
    assert stored.embedding_profile == NewProvider.profile
    assert stored.embedding_revision == NewProvider.revision
    assert [(r.id, r.text, r.page_number, r.chunk_order) for r in db.scalars(select(ChunkRow).order_by(ChunkRow.id))] == old_chunks
    db.commit()
    previous_vectors = [list(r.embedding) for r in db.scalars(select(ChunkRow).order_by(ChunkRow.id))]
    db.commit()
    class FailingProvider(NewProvider):
        profile = "c" * 64
        calls = 0
        def embed_documents(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("Failure after first document was converted")
            return [[1.0] + [0.0] * 383 for text in texts]
    with pytest.raises(RuntimeError):
        reembed_corpus(db, FailingProvider())
    assert db.get(EmbeddingCorpusRow, 1).profile == NewProvider.profile
    assert [list(r.embedding) for r in db.scalars(select(ChunkRow).order_by(ChunkRow.id))] == previous_vectors


@pytest.mark.semantic
@pytest.mark.skipif(os.getenv("RUN_SEMANTIC_TESTS") != "1", reason="Set RUN_SEMANTIC_TESTS=1 for real MiniLM")
def test_real_semantic_pdf_upload(db, pdf_bytes):
    app = create_app(Settings(embedding_provider="sentence_transformers", embedding_model="sentence-transformers/all-MiniLM-L6-v2"))
    app.dependency_overrides[get_session] = lambda: db
    with TestClient(app) as client:
        response = client.post("/api/documents", files={"file": ("semantic.pdf", pdf_bytes(["Physical exercise improves cardiovascular health."]), "application/pdf")})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert len(result["embedding_revision"]) == 40
    assert len(result["embedding_profile"]) == 64
    rows = list(db.scalars(select(ChunkRow).where(ChunkRow.document_id == result["id"])))
    assert len(rows) == 1 and len(rows[0].embedding) == 384


@pytest.mark.parametrize("models", [["local-hash-v1"], ["local-hash-v1", "unknown-model"]])
def test_profile_migration_preserves_and_classifies_legacy_data(db, models):
    import importlib.util
    from pathlib import Path
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from app.domain.embeddings import LEXICAL_PROFILE
    schema = "test_profiles_" + uuid4().hex
    db.execute(text(f'CREATE SCHEMA "{schema}"'))
    db.execute(text(f'SET LOCAL search_path TO "{schema}", public'))
    db.execute(text("CREATE TABLE documents (id integer PRIMARY KEY, embedding_model varchar(100) NOT NULL, embedding_dimension integer NOT NULL, preserved_text text NOT NULL)"))
    for number, model in enumerate(models):
        db.execute(text("INSERT INTO documents VALUES (:id, :model, 384, 'Preserve this text')"), {"id": number, "model": model})
    path = Path(__file__).resolve().parents[1] / "migrations/versions/0003_embedding_profiles.py"
    spec = importlib.util.spec_from_file_location("profile_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with Operations.context(MigrationContext.configure(db.connection())):
        module.upgrade()
    rows = list(db.execute(text("SELECT embedding_model, embedding_profile, preserved_text FROM documents ORDER BY id")))
    assert [row[0] for row in rows] == models
    assert all(row[2] == "Preserve this text" for row in rows)
    profile = db.scalar(text("SELECT profile FROM embedding_corpus"))
    assert (profile == LEXICAL_PROFILE) == (models == ["local-hash-v1"])
    # Fixture rollback also removes this isolated test schema.
