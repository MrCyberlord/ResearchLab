"""Cosine search and RAG against PostgreSQL; all writes use private rollback schemas."""
import hashlib
import json
import math
import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from test_ingestion_db import db  # Reuse the established private-schema fixture.
from app.api.dependencies import get_rag_service, get_retrieval_service, get_session
from app.application.ingestion import IngestionService
from app.application.rag import RagService
from app.application.retrieval import RetrievalService
from app.domain.documents import Chunk, Document, IncompatibleEmbeddings
from app.domain.rag import LLMResult, RetrievalUnavailable, SourceReference, TokenUsage
from app.infrastructure.config import Settings
from app.infrastructure.document_models import ChunkRow
from app.infrastructure.document_repository import SQLAlchemyDocumentRepository
from app.infrastructure.embeddings import LocalHashEmbeddings, create_embedding_provider
from app.infrastructure.pdf import IsolatedPDFExtractor
from app.infrastructure.retrieval_repository import SQLAlchemyRetrievalRepository
from app.main import create_app

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.getenv("RUN_DB_TESTS") != "1", reason="Set RUN_DB_TESTS=1 to use migrated local PostgreSQL")]


def vector(x=1.0, y=0.0):
    norm = math.sqrt(x*x + y*y)
    return [x/norm, y/norm] + [0.0] * 382


class FixedEmbeddings(LocalHashEmbeddings):
    def embed_query(self, text):
        return vector()


class EvidenceLLM:
    provider = "test-provider"
    model = "test-model"

    def __init__(self):
        self.calls = []

    def generate(self, system_prompt, user_prompt, source_ids):
        self.calls.append((system_prompt, user_prompt, source_ids))
        source = json.loads(user_prompt)["sources"][0]
        return LLMResult(answer="The evidence is in the supplied document [S1].",
                         evidence=[SourceReference(source["source_id"], source["text"])],
                         insufficient_evidence=False, usage=TokenUsage(40, 12))


def seed(db, filename, texts_and_vectors, provider=None):
    provider = provider or FixedEmbeddings()
    document_id = uuid4()
    content_hash = hashlib.sha256(document_id.bytes).hexdigest()
    document = Document(document_id, filename, content_hash, 100, 2, len(texts_and_vectors),
                        provider.model, provider.dimension, 1000, 150, provider.profile, provider.revision)
    chunks = [Chunk(document_id, page, order, content)
              for order, (page, content, _) in enumerate(texts_and_vectors)]
    repository = SQLAlchemyDocumentRepository(db)
    with repository.transaction():
        repository.bind_profile(provider.profile)
        repository.add(document, chunks, [embedding for _, _, embedding in texts_and_vectors])
    return document


def test_pgvector_uses_cosine_distance_and_top_k(db):
    best = seed(db, "best.pdf", [(2, "The kettle warranty lasts three years.", vector())])
    runner_up = seed(db, "other.pdf", [(1, "A related warranty passage.", vector(.8, .6))])
    seed(db, "unrelated.pdf", [(1, "An unrelated gardening passage.", vector(0, 1))])
    repository = SQLAlchemyRetrievalRepository(db)
    results = repository.search(vector(), FixedEmbeddings.profile, 2, .25, None)
    assert [match.document_id for match in results] == [best.id, runner_up.id]
    assert results[0].distance == pytest.approx(0, abs=1e-6)
    assert results[1].distance == pytest.approx(.2, abs=1e-5)  # Cosine, not Euclidean distance.
    assert results[0].page_number == 2
    assert len(repository.search(vector(), FixedEmbeddings.profile, 1, .25, None)) == 1
    assert [match.document_id for match in repository.search(vector(), FixedEmbeddings.profile, 20, .95, None)] == [best.id]


def test_document_filter_is_applied_before_top_k_and_never_broadens(db):
    first = seed(db, "first.pdf", [(1, "First document.", vector())])
    second = seed(db, "second.pdf", [(2, "Selected document.", vector(.8, .6))])
    repository = SQLAlchemyRetrievalRepository(db)
    selected = repository.search(vector(), FixedEmbeddings.profile, 1, .25, [second.id])
    assert len(selected) == 1 and selected[0].document_id == second.id
    assert selected[0].filename == "second.pdf"
    assert all(item.document_id != first.id for item in selected)
    assert repository.search(vector(), FixedEmbeddings.profile, 5, .25, []) == []
    assert repository.search(vector(), FixedEmbeddings.profile, 5, .25, [uuid4()]) == []


def test_postgresql_keyword_search_ranks_terms_safely_and_applies_document_filter(db):
    best = seed(db, "best-keyword.pdf", [(1, "Quasar calibration repeats quasar calibration checks.", vector())])
    other = seed(db, "other-keyword.pdf", [(1, "A calibration procedure for the quasar sensor.", vector(.8, .6))])
    unrelated = seed(db, "garden.pdf", [(1, "Tomato seedlings need warm soil.", vector(0, 1))])
    repository = SQLAlchemyRetrievalRepository(db)
    matches = repository.keyword_search("quasar calibration", vector(), FixedEmbeddings.profile, 5, None)
    assert [item.document_id for item in matches[:2]] == [best.id, other.id]
    assert all(item.document_id != unrelated.id for item in matches)
    assert matches[0].distance == pytest.approx(0, abs=1e-6)
    selected = repository.keyword_search("quasar calibration", vector(), FixedEmbeddings.profile, 5, [other.id])
    assert [item.document_id for item in selected] == [other.id]
    assert repository.keyword_search("quasar calibration", vector(), FixedEmbeddings.profile, 5, []) == []
    # websearch_to_tsquery accepts raw user syntax without SQL/query-parser errors.
    assert repository.keyword_search('""" ((( OR -', vector(), FixedEmbeddings.profile, 5, None) == []


def test_keyword_search_vector_has_gin_index_in_isolated_schema(db):
    definition = db.scalar(text("""
        SELECT indexdef FROM pg_indexes
        WHERE schemaname = current_schema() AND indexname = 'ix_chunks_search_vector'
    """))
    assert definition and "USING gin" in definition and "search_vector" in definition


def test_equal_distance_results_have_stable_chunk_order(db):
    document = seed(db, "ties.pdf", [(1, "One.", vector()), (1, "Two.", vector()), (2, "Three.", vector())])
    repository = SQLAlchemyRetrievalRepository(db)
    first = repository.search(vector(), FixedEmbeddings.profile, 3, .25, [document.id])
    second = repository.search(vector(), FixedEmbeddings.profile, 3, .25, [document.id])
    assert [item.chunk_order for item in first] == [0, 1, 2]
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]


def test_repository_rejects_incompatible_embeddings_on_direct_search(db):
    seed(db, "legacy.pdf", [(1, "Legacy lexical vectors.", vector())])
    repository = SQLAlchemyRetrievalRepository(db)
    with pytest.raises(IncompatibleEmbeddings):
        repository.check_profile("different-model-profile")
    with pytest.raises(IncompatibleEmbeddings):
        repository.search(vector(), "different-model-profile", 5, .25, None)


def test_empty_store_does_not_generate_an_answer(db):
    llm = EvidenceLLM()
    service = RagService(RetrievalService(SQLAlchemyRetrievalRepository(db), FixedEmbeddings()), llm)
    result = service.answer("What is the warranty?")
    assert result.status == "insufficient_evidence" and result.answer is None
    assert result.citations == [] and result.retrieved_chunk_count == 0
    assert llm.calls == []


def test_rag_citation_metadata_matches_the_persisted_supporting_chunk(db):
    document = seed(db, "actual-warranty.pdf", [(2, "The Astral kettle warranty lasts three years.", vector())])
    result = RagService(RetrievalService(SQLAlchemyRetrievalRepository(db), FixedEmbeddings()), EvidenceLLM()).answer("How long is the warranty?")
    assert result.status == "answered" and len(result.citations) == 1
    citation = result.citations[0]
    persisted = db.get(ChunkRow, citation.chunk_id)
    assert citation.document_id == document.id == persisted.document_id
    assert citation.filename == "actual-warranty.pdf"
    assert citation.page_number == persisted.page_number == 2
    assert citation.chunk_order == persisted.chunk_order == 0
    assert citation.quote in persisted.text
    assert citation.text == persisted.text


def test_document_listing_and_filtered_http_questions(db):
    first = seed(db, "first.pdf", [(1, "First document evidence.", vector())])
    second = seed(db, "selected.pdf", [(2, "Selected document evidence.", vector())])
    retrieval = RetrievalService(SQLAlchemyRetrievalRepository(db), FixedEmbeddings())
    app = create_app(Settings())
    app.dependency_overrides[get_session] = lambda: db
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval
    app.dependency_overrides[get_rag_service] = lambda: RagService(retrieval, EvidenceLLM())
    with TestClient(app) as client:
        listing = client.get("/api/documents", params={"limit": 1, "offset": 0})
        assert listing.status_code == 200, listing.text
        assert listing.json()["total"] == 2 and len(listing.json()["documents"]) == 1
        following = client.get("/api/documents", params={"limit": 1, "offset": 1})
        assert following.status_code == 200
        assert following.json()["documents"][0]["id"] != listing.json()["documents"][0]["id"]
        response = client.post("/api/retrieval", json={"query": "Evidence?", "document_ids": [str(second.id)]})
        assert response.status_code == 200, response.text
        assert [item["document_id"] for item in response.json()["chunks"]] == [str(second.id)]
        hybrid = client.post("/api/retrieval", json={"query": "Selected document evidence",
                             "strategy": "hybrid", "document_ids": [str(second.id)]})
        assert hybrid.status_code == 200, hybrid.text
        assert hybrid.json()["strategy"] == "hybrid"
        assert {item["document_id"] for item in hybrid.json()["chunks"]} == {str(second.id)}
        answer = client.post("/api/rag/questions", json={"question": "Evidence?", "document_ids": [str(second.id)]})
        assert answer.status_code == 200, answer.text
        assert {citation["document_id"] for citation in answer.json()["citations"]} == {str(second.id)}
        assert all(citation["document_id"] != str(first.id) for citation in answer.json()["citations"])
        hybrid_answer = client.post("/api/rag/questions", json={"question": "Selected document evidence",
                                    "strategy": "hybrid", "document_ids": [str(second.id)]})
        assert hybrid_answer.status_code == 200, hybrid_answer.text
        assert hybrid_answer.json()["retrieval_strategy"] == "hybrid"
        assert {citation["document_id"] for citation in hybrid_answer.json()["citations"]} == {str(second.id)}
        assert client.get("/api/documents", params={"limit": 101}).status_code == 422


def test_database_failures_are_reported_without_sensitive_query_content(db):
    seed(db, "private.pdf", [(1, "Private text must not appear in an error.", vector())])
    db.execute(text("ALTER TABLE chunks RENAME COLUMN embedding TO temporarily_unavailable_embedding"))
    db.commit()
    with pytest.raises(RetrievalUnavailable) as error:
        SQLAlchemyRetrievalRepository(db).search(vector(), FixedEmbeddings.profile, 5, .25, None)
    assert "Private text" not in str(error.value)
    assert "SELECT" not in str(error.value)


@pytest.mark.semantic
@pytest.mark.skipif(os.getenv("RUN_SEMANTIC_TESTS") != "1", reason="Set RUN_SEMANTIC_TESTS=1 for genuine MiniLM")
def test_genuine_minilm_retrieval_uses_ingestion_profile_and_document_isolation(db, pdf_bytes):
    settings = Settings(embedding_provider="sentence_transformers", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = create_embedding_provider(settings)
    service = IngestionService(SQLAlchemyDocumentRepository(db), IsolatedPDFExtractor(10, 10000, 15, 256), embeddings, 1000, 150)
    relevant = service.ingest(pdf_bytes(["Reset a forgotten account password with the recovery link sent by email."]), "account.pdf")
    unrelated = service.ingest(pdf_bytes(["Tomato plants grow best in rich soil under warm sunlight."]), "garden.pdf")
    retrieval = RetrievalService(SQLAlchemyRetrievalRepository(db), embeddings, min_similarity=-1)
    result = retrieval.retrieve("How do I recover access when I forget my password?", top_k=2)
    assert result.embedding_profile == relevant.embedding_profile == embeddings.profile
    assert result.chunks[0].document_id == relevant.id
    assert result.chunks[0].distance < result.chunks[1].distance
    selected = retrieval.retrieve("How do I recover my password?", document_ids=[unrelated.id])
    assert len(selected.chunks) == 1 and selected.chunks[0].document_id == unrelated.id


@pytest.mark.llm_smoke
@pytest.mark.semantic
@pytest.mark.skipif(os.getenv("RUN_LLM_SMOKE") != "1" or os.getenv("RUN_SEMANTIC_TESTS") != "1",
                    reason="Set RUN_LLM_SMOKE=1 and RUN_SEMANTIC_TESTS=1 for the real complete RAG pipeline")
def test_real_rag_pipeline_ingestion_retrieval_answer_and_citations(db, pdf_bytes):
    """Synthetic PDF only; genuine MiniLM, pgvector and configured local generation."""
    import re
    import unicodedata
    from app.infrastructure.llm import create_llm_provider

    settings = Settings(embedding_provider="sentence_transformers",
                        embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = create_embedding_provider(settings)
    ingestion = IngestionService(SQLAlchemyDocumentRepository(db),
                                 IsolatedPDFExtractor(10, 10000, 15, 256), embeddings, 1000, 150)
    fact = "The Astral kettle warranty lasts three years from the purchase date. The warranty covers manufacturing defects."
    document = ingestion.ingest(pdf_bytes(["Astral product support handbook.", fact]), "astral-e2e.pdf")
    retrieval = RetrievalService(SQLAlchemyRetrievalRepository(db), embeddings,
                                 default_top_k=1, min_similarity=settings.retrieval_min_similarity)
    service = RagService(retrieval, create_llm_provider(settings),
                         context_max_bytes=settings.rag_max_context_bytes,
                         prompt_max_bytes=settings.rag_max_prompt_bytes)
    result = service.answer("For how many years is an Astral kettle covered after purchase?",
                            top_k=1, document_ids=[document.id])
    assert result.status == "answered" and result.answer
    assert "three" in result.answer.lower() or re.search(r"\b3\b", result.answer)
    assert result.embedding_profile == document.embedding_profile == embeddings.profile
    assert result.llm_provider == settings.llm_provider and result.llm_model == settings.llm_model
    assert result.retrieved_chunk_count == result.context_chunk_count == 1
    assert result.citations
    canonical = lambda value: " ".join(unicodedata.normalize("NFKC", value).split())
    for citation in result.citations:
        stored = db.get(ChunkRow, citation.chunk_id)
        assert stored is not None and stored.document_id == document.id
        assert citation.document_id == document.id
        assert citation.filename == "astral-e2e.pdf"
        assert citation.page_number == stored.page_number == 2
        assert citation.chunk_order == stored.chunk_order == 1
        assert citation.quote.strip() and canonical(citation.quote) in canonical(stored.text)
        assert citation.text == stored.text
    if result.usage:
        if result.usage.prompt_tokens is not None:
            assert result.usage.prompt_tokens > 0
        if result.usage.completion_tokens is not None:
            assert result.usage.completion_tokens > 0
