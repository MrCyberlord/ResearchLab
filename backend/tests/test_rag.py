"""Application and HTTP contracts; no external model/provider calls."""
import json
import math
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_rag_service, get_retrieval_service
from app.application.rag import RagService
from app.application.retrieval import RetrievalService
from app.domain.documents import EmbeddingFailure, IncompatibleEmbeddings
from app.domain.embeddings import LEXICAL_PROFILE
from app.domain.rag import (
    LLMInvalidResponse, LLMResult, LLMTimeout, LLMUnavailable, QuestionInvalid,
    RerankerUnavailable, RetrievedChunk, SourceReference, TokenUsage,
)
from app.infrastructure.config import Settings
from app.infrastructure.embeddings import LocalHashEmbeddings
from app.main import create_app


class FixedEmbeddings(LocalHashEmbeddings):
    def __init__(self, vector=None):
        self.vector = [1.0] + [0.0] * 383 if vector is None else vector
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return self.vector

    def embed_documents(self, texts):
        raise AssertionError("Retrieval must use the query embedding contract")


class MemoryRetrievalRepository:
    def __init__(self, chunks=(), profile=LEXICAL_PROFILE):
        self.chunks = list(chunks)
        self.profile = profile
        self.searches = []

    def check_profile(self, profile):
        if self.profile != profile:
            raise IncompatibleEmbeddings()

    def search(self, vector, profile, top_k, min_similarity, document_ids):
        self.check_profile(profile)
        self.searches.append((vector, profile, top_k, min_similarity, document_ids))
        chunks = self.chunks
        if document_ids is not None:
            chunks = [chunk for chunk in chunks if chunk.document_id in document_ids]
        return [chunk for chunk in chunks if 1 - chunk.distance >= min_similarity][:top_k]

    def list_documents(self, limit, offset):
        return [], 0


class RecordingLLM:
    provider = "test-provider"
    model = "test-model"

    def __init__(self, result=None, failure=None):
        self.result = result
        self.failure = failure
        self.calls = []

    def generate(self, system_prompt, user_prompt, source_ids):
        self.calls.append((system_prompt, user_prompt, source_ids))
        if self.failure:
            raise self.failure
        if self.result is not None:
            return self.result
        source = json.loads(user_prompt)["sources"][0]
        return LLMResult(
            answer="The warranty lasts three years [S1].",
            evidence=[SourceReference(source["source_id"], source["text"])],
            insufficient_evidence=False,
            usage=TokenUsage(prompt_tokens=90, completion_tokens=18),
        )


def chunk(text="The Astral kettle warranty lasts three years.", *, chunk_id=23, document_id=None,
          filename="warranty.pdf", page_number=2, chunk_order=4, distance=0.1):
    return RetrievedChunk(chunk_id=chunk_id, document_id=document_id or uuid4(), filename=filename,
                          page_number=page_number, chunk_order=chunk_order, text=text, distance=distance)


def retrieval(chunks=(), embeddings=None, repository=None):
    return RetrievalService(repository or MemoryRetrievalRepository(chunks), embeddings or FixedEmbeddings())


def rag(chunks=(), llm=None, **kwargs):
    return RagService(retrieval(chunks), llm or RecordingLLM(), **kwargs)


@pytest.mark.parametrize("question", ["", " \t\n ", "\x00", "x" * 2001, "\U0001f600" * 1500])
def test_invalid_question_is_rejected_before_embedding(question):
    embeddings = FixedEmbeddings()
    with pytest.raises(QuestionInvalid):
        retrieval(embeddings=embeddings).retrieve(question)
    assert embeddings.queries == []


@pytest.mark.parametrize("top_k", [0, -1, 21])
def test_retrieval_rejects_out_of_range_top_k(top_k):
    with pytest.raises(QuestionInvalid):
        retrieval([chunk()]).retrieve("How long is the warranty?", top_k=top_k)


def test_query_embedding_and_document_filter_are_forwarded():
    document_id = uuid4()
    match = chunk(document_id=document_id)
    embeddings = FixedEmbeddings()
    repository = MemoryRetrievalRepository([match, chunk()])
    result = retrieval(embeddings=embeddings, repository=repository).retrieve(
        "How long is the warranty?", top_k=2, document_ids=[document_id])
    assert embeddings.queries == ["How long is the warranty?"]
    assert result.chunks == [match]
    assert repository.searches[0][2] == 2
    assert repository.searches[0][4] == [document_id]
    assert result.embedding_profile == embeddings.profile
    assert result.retrieval_ms >= 0


def test_empty_document_filter_never_broadens_to_entire_corpus():
    embeddings = FixedEmbeddings()
    repository = MemoryRetrievalRepository([chunk()])
    result = retrieval(embeddings=embeddings, repository=repository).retrieve(
        "How long is the warranty?", document_ids=[])
    assert result.chunks == []
    assert embeddings.queries == []
    assert repository.searches == []


def test_incompatible_profile_is_checked_before_query_embedding():
    embeddings = FixedEmbeddings()
    repository = MemoryRetrievalRepository([chunk()], profile="different-model-profile")
    with pytest.raises(IncompatibleEmbeddings):
        retrieval(embeddings=embeddings, repository=repository).retrieve("Warranty length?")
    assert embeddings.queries == []
    assert repository.searches == []


@pytest.mark.parametrize("vector", [[1.0] * 128, [0.0] * 384,
                                    [float("nan")] * 384, [float("inf")] * 384,
                                    [1.0] * 384])
def test_invalid_query_vectors_are_not_sent_to_database(vector):
    repository = MemoryRetrievalRepository([chunk()])
    with pytest.raises(EmbeddingFailure):
        retrieval(embeddings=FixedEmbeddings(vector), repository=repository).retrieve("Warranty?")
    assert repository.searches == []


def test_embedding_failure_never_invokes_llm():
    class BrokenEmbeddings(FixedEmbeddings):
        def embed_query(self, text): raise RuntimeError("private provider diagnostic")
    llm = RecordingLLM()
    service = RagService(retrieval([chunk()], embeddings=BrokenEmbeddings()), llm)
    with pytest.raises(EmbeddingFailure):
        service.answer("Warranty?")
    assert llm.calls == []


def test_answer_citations_map_only_to_supplied_database_evidence():
    supporting_chunk = chunk()
    llm = RecordingLLM()
    result = rag([supporting_chunk], llm).answer("How long is the kettle warranty?")
    assert result.status == "answered"
    assert "three" in result.answer
    assert result.message is None
    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.source_id == "S1"
    assert citation.document_id == supporting_chunk.document_id
    assert citation.filename == supporting_chunk.filename
    assert citation.page_number == 2
    assert citation.chunk_id == 23 and citation.chunk_order == 4
    assert citation.text == supporting_chunk.text
    assert citation.quote in supporting_chunk.text
    assert result.retrieved_chunk_count == result.context_chunk_count == 1
    assert result.llm_provider == "test-provider" and result.llm_model == "test-model"
    assert result.usage.prompt_tokens == 90 and result.usage.completion_tokens == 18
    assert result.retrieval_ms >= 0 and result.generation_ms >= 0
    system, user, source_ids = llm.calls[0]
    assert source_ids == ["S1"]
    assert json.loads(user)["question"] == "How long is the kettle warranty?"
    assert json.loads(user)["sources"] == [{"source_id": "S1", "text": supporting_chunk.text}]
    assert "warranty.pdf" not in user  # Metadata is not model-authored.
    assert "insufficient" in system.lower()


def test_context_and_total_prompt_are_bounded_in_utf8_bytes():
    llm = RecordingLLM()
    chunks = [chunk("The warranty lasts three years. " * 500, chunk_id=index + 1) for index in range(5)]
    result = rag(chunks, llm, context_max_bytes=700, prompt_max_bytes=3000).answer("Warranty length?")
    system, user, _ = llm.calls[0]
    sources = json.loads(user)["sources"]
    assert sum(len(source["text"].encode("utf-8")) for source in sources) <= 700
    assert len((system + user).encode("utf-8")) <= 3000
    assert result.context_chunk_count == len(sources)
    assert result.context_chunk_count <= result.retrieved_chunk_count
    assert all(citation.text == sources[int(citation.source_id[1:])-1]["text"] for citation in result.citations)


def test_no_retrieved_evidence_does_not_call_generation():
    llm = RecordingLLM()
    result = rag([], llm).answer("What is the warranty?")
    assert result.status == "insufficient_evidence"
    assert result.answer is None and result.citations == []
    assert result.message
    assert result.generation_ms == 0
    assert llm.calls == []


def test_model_insufficient_evidence_is_explicit_and_has_no_fake_citations():
    llm = RecordingLLM(LLMResult(answer="", evidence=[], insufficient_evidence=True,
                               usage=TokenUsage(prompt_tokens=50, completion_tokens=10)))
    result = rag([chunk()], llm).answer("Who designed the kettle?")
    assert result.status == "insufficient_evidence"
    assert result.answer is None and result.citations == []
    assert result.message
    assert len(llm.calls) == 1
    assert result.usage.completion_tokens == 10


@pytest.mark.parametrize("answer,evidence", [
    ("Three years [S99].", [SourceReference("S99", "The Astral kettle warranty lasts three years.")]),
    ("Three years [S1].", [SourceReference("S1", "A fabricated quote not present in evidence.")]),
    ("Three years.", []),
    ("Three years [S999].", [SourceReference("S1", "The Astral kettle warranty lasts three years.")]),
])
def test_fabricated_or_missing_citations_are_rejected(answer, evidence):
    llm = RecordingLLM(LLMResult(answer=answer, evidence=evidence, insufficient_evidence=False))
    with pytest.raises(LLMInvalidResponse):
        rag([chunk()], llm).answer("Warranty?")


def test_citation_quote_comparison_accepts_equivalent_whitespace():
    llm = RecordingLLM(LLMResult(answer="Three years [S1].", evidence=[
        SourceReference("S1", "The Astral kettle warranty lasts three years.")], insufficient_evidence=False))
    result = rag([chunk("The Astral kettle\nwarranty lasts   three years.")], llm).answer("Warranty?")
    assert result.status == "answered"
    assert result.citations[0].page_number == 2


def test_invalid_quote_gets_one_strict_deadline_bounded_regeneration():
    class RepairingLLM:
        provider = "test-provider"
        model = "test-model"

        def __init__(self):
            self.calls = []

        def generate(self, system_prompt, user_prompt, source_ids, *, timeout_seconds=None):
            self.calls.append((system_prompt, timeout_seconds))
            if len(self.calls) == 1:
                return LLMResult(
                    answer="Shutters close after sunset [S1].",
                    evidence=[SourceReference(
                        "S1", "the shutters close automatically after sunset."
                    )],
                    insufficient_evidence=False,
                    usage=TokenUsage(prompt_tokens=10, completion_tokens=4),
                )
            return LLMResult(
                answer="Shutters close after sunset [S1].",
                evidence=[SourceReference(
                    "S1", "The shutters close automatically after sunset."
                )],
                insufficient_evidence=False,
                usage=TokenUsage(prompt_tokens=12, completion_tokens=5),
            )

    llm = RepairingLLM()
    result = rag([
        chunk("The shutters close automatically after sunset.")
    ], llm).grounding.answer(
        "What happens after sunset?",
        [chunk("The shutters close automatically after sunset.")],
        timeout_seconds=2,
    )
    assert result.status == "answered"
    assert result.citations[0].quote.startswith("The shutters")
    assert len(llm.calls) == 2
    assert "character-for-character" in llm.calls[1][0]
    assert 0 < llm.calls[1][1] <= llm.calls[0][1] <= 2
    assert result.usage.prompt_tokens == 22
    assert result.usage.completion_tokens == 9


def test_invalid_quote_regeneration_is_bounded_to_one_retry():
    llm = RecordingLLM(LLMResult(
        answer="Three years [S1].",
        evidence=[SourceReference("S1", "a fabricated quote")],
        insufficient_evidence=False,
    ))
    with pytest.raises(LLMInvalidResponse):
        rag([chunk()], llm).answer("Warranty?")
    assert len(llm.calls) == 2


@pytest.mark.parametrize("failure", [LLMUnavailable(), LLMTimeout(), LLMInvalidResponse()])
def test_provider_failures_propagate_without_a_fake_answer(failure):
    llm = RecordingLLM(failure=failure)
    with pytest.raises(type(failure)):
        rag([chunk()], llm).answer("Warranty?")
    assert len(llm.calls) == 1


@pytest.fixture
def client():
    app = create_app(Settings(_env_file=None, database_url=None))
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval([chunk()])
    app.dependency_overrides[get_rag_service] = lambda: rag([chunk()])
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("path,payload", [
    ("/api/retrieval", {"query": ""}),
    ("/api/retrieval", {"query": "x" * 2001}),
    ("/api/retrieval", {"query": "valid", "top_k": 0}),
    ("/api/retrieval", {"query": "valid", "top_k": 21}),
    ("/api/retrieval", {"query": "valid", "document_ids": ["not-a-uuid"]}),
    ("/api/retrieval", {"query": "valid", "strategy": "lexical"}),
    ("/api/rag/questions", {"question": " \t "}),
    ("/api/rag/questions", {"question": "x" * 2001}),
    ("/api/rag/questions", {"question": "valid", "document_ids": ["not-a-uuid"]}),
])
def test_api_request_validation(client, path, payload):
    assert client.post(path, json=payload).status_code == 422


def test_api_returns_retrieval_and_grounded_citations(client):
    retrieval_response = client.post("/api/retrieval", json={"query": "Warranty?"})
    assert retrieval_response.status_code == 200, retrieval_response.text
    assert len(retrieval_response.json()["chunks"]) == 1
    assert retrieval_response.json()["strategy"] == "semantic"
    response = client.post("/api/rag/questions", json={"question": "Warranty?"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "answered"
    assert result["citations"][0]["filename"] == "warranty.pdf"
    assert result["citations"][0]["page_number"] == 2
    assert result["citations"][0]["chunk_id"] == 23
    assert result["usage"]["prompt_tokens"] == 90
    assert result["retrieval_strategy"] == "semantic"


def test_api_empty_selection_is_insufficient_evidence(client):
    response = client.post("/api/rag/questions", json={"question": "Warranty?", "document_ids": []})
    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"
    assert response.json()["citations"] == []


@pytest.mark.parametrize("failure,status", [
    (IncompatibleEmbeddings(), 409), (EmbeddingFailure("Safe embedding failure"), 503),
    (RerankerUnavailable(), 503),
    (LLMUnavailable(), 503), (LLMTimeout(), 504), (LLMInvalidResponse(), 502),
])
def test_api_reports_pipeline_failures(client, failure, status):
    class BrokenService:
        def answer(self, *args, **kwargs): raise failure
    client.app.dependency_overrides[get_rag_service] = lambda: BrokenService()
    response = client.post("/api/rag/questions", json={"question": "Warranty?"})
    assert response.status_code == status, response.text
    assert response.json()["detail"]["message"]


def test_rag_configuration_is_available_without_database_or_provider_calls(client):
    response = client.get("/api/rag/config")
    assert response.status_code == 200
    config = response.json()
    assert config["default_top_k"] == 5 and config["max_top_k"] == 20
    assert config["default_strategy"] == "semantic"
    assert config["available_strategies"] == ["semantic", "hybrid", "hybrid_rerank"]
    assert config["reranker_model"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert config["llm_provider"] == "ollama"
    assert config["max_question_chars"] == 2000


def test_structured_metric_logging_excludes_sensitive_fields():
    import logging
    from app.infrastructure.logging import JsonFormatter
    record = logging.LogRecord("researchlab", logging.INFO, __file__, 1, "rag_completed", (), None)
    record.retrieval_ms = 3.5
    record.generation_ms = 15.2
    record.retrieved_chunk_count = 3
    record.context_chunk_count = 2
    record.prompt_tokens = 120
    record.completion_tokens = 20
    record.question = "sensitive question"
    record.context = "sensitive document text"
    record.api_key = "sensitive key"
    result = json.loads(JsonFormatter().format(record))
    assert result["retrieval_ms"] == 3.5 and result["generation_ms"] == 15.2
    assert result["retrieved_chunk_count"] == 3 and result["context_chunk_count"] == 2
    assert result["prompt_tokens"] == 120 and result["completion_tokens"] == 20
    assert not {"question", "context", "api_key"} & set(result)
