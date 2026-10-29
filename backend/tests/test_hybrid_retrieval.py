import os
from uuid import uuid4

import numpy as np
import pytest

from app.application.retrieval import RetrievalService, reciprocal_rank_fusion
from app.domain.documents import IncompatibleEmbeddings
from app.domain.embeddings import LEXICAL_PROFILE
from app.domain.rag import RerankerUnavailable, RetrievedChunk
from app.infrastructure.embeddings import LocalHashEmbeddings
from app.infrastructure.config import Settings
from app.infrastructure.reranker import LocalCrossEncoderReranker, create_reranker_provider


def chunk(chunk_id, text=None, distance=.2, document_id=None, chunk_order=None):
    return RetrievedChunk(
        chunk_id, document_id or uuid4(), f"document-{chunk_id}.txt", 1,
        chunk_id if chunk_order is None else chunk_order, text or f"passage {chunk_id}", distance,
    )


class FixedEmbeddings(LocalHashEmbeddings):
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0] + [0.0] * 383


class StrategyRepository:
    def __init__(self, semantic=(), keyword=(), profile=LEXICAL_PROFILE):
        self.semantic = list(semantic)
        self.keyword = list(keyword)
        self.profile = profile
        self.semantic_calls = []
        self.keyword_calls = []

    def check_profile(self, profile):
        if profile != self.profile:
            raise IncompatibleEmbeddings()

    def search(self, vector, profile, top_k, min_similarity, document_ids):
        self.check_profile(profile)
        self.semantic_calls.append((top_k, min_similarity, document_ids))
        values = self.semantic
        if document_ids is not None:
            values = [item for item in values if item.document_id in document_ids]
        return values[:top_k]

    def keyword_search(self, query, vector, profile, top_k, document_ids):
        self.check_profile(profile)
        self.keyword_calls.append((query, top_k, document_ids))
        values = self.keyword
        if document_ids is not None:
            values = [item for item in values if item.document_id in document_ids]
        return values[:top_k]

    def list_documents(self, limit, offset):
        return [], 0


class FixedReranker:
    provider = "fixed"
    model = "fixed-reranker"
    revision = "a" * 40

    def __init__(self, scores=None, failure=None):
        self.scores = scores
        self.failure = failure
        self.calls = []

    def score(self, query, passages):
        self.calls.append((query, passages))
        if self.failure:
            raise self.failure
        return self.scores if self.scores is not None else list(reversed(range(len(passages))))


def service(repository, reranker=None):
    return RetrievalService(
        repository, FixedEmbeddings(), min_similarity=-1, semantic_candidates=20,
        keyword_candidates=20, rrf_constant=60, reranker=reranker, reranker_candidates=20,
    )


def test_rrf_orders_by_combined_rank_and_deduplicates_stable_chunk_identity():
    first, second, third = chunk(1), chunk(2), chunk(3)
    result = reciprocal_rank_fusion([[first, second, first], [second, third]], 60, 10)
    assert [item.chunk_id for item in result] == [2, 1, 3]
    assert len({item.chunk_id for item in result}) == len(result)


def test_hybrid_fuses_semantic_and_keyword_candidates_and_preserves_true_distance():
    first, second, third = chunk(1, distance=.1), chunk(2, distance=.8), chunk(3, distance=.4)
    repository = StrategyRepository([first, third], [second, first])
    result = service(repository).retrieve("thermal shutters", top_k=3, strategy="hybrid")
    assert [item.chunk_id for item in result.chunks] == [1, 2, 3]
    assert [item.distance for item in result.chunks] == [.1, .8, .4]
    assert result.strategy == "hybrid"
    assert repository.semantic_calls == [(20, -1, None)]
    assert repository.keyword_calls == [("thermal shutters", 20, None)]


def test_empty_keyword_results_leave_semantic_order_and_document_filter_is_forwarded():
    selected, excluded = uuid4(), uuid4()
    first = chunk(1, document_id=selected)
    second = chunk(2, document_id=excluded)
    repository = StrategyRepository([first, second], [])
    result = service(repository).retrieve("query", top_k=2, document_ids=[selected], strategy="hybrid")
    assert result.chunks == [first]
    assert repository.semantic_calls[0][2] == [selected]
    assert repository.keyword_calls[0][2] == [selected]


def test_hybrid_rerank_reorders_only_fused_candidates_and_is_deterministic_on_ties():
    document_id = uuid4()
    first = chunk(1, document_id=document_id, chunk_order=0)
    second = chunk(2, document_id=document_id, chunk_order=1)
    third = chunk(3, document_id=document_id, chunk_order=2)
    reranker = FixedReranker([.1, .9, .9])
    result = service(StrategyRepository([first, second, third], [first, second, third]), reranker).retrieve(
        "power transfer", top_k=3, strategy="hybrid_rerank"
    )
    assert [item.chunk_id for item in result.chunks] == [2, 3, 1]
    assert reranker.calls == [("power transfer", [first.text, second.text, third.text])]
    assert result.strategy == "hybrid_rerank"


@pytest.mark.parametrize("scores", [[1.0], [float("nan"), 1.0]])
def test_invalid_reranker_scores_fail_closed(scores):
    reranker = FixedReranker(scores)
    with pytest.raises(RerankerUnavailable):
        service(StrategyRepository([chunk(1), chunk(2)], []), reranker).retrieve(
            "query", top_k=2, strategy="hybrid_rerank"
        )


def test_reranker_provider_failure_is_safe_and_no_fallback_answer_is_invented():
    private = RuntimeError("private model path")
    with pytest.raises(RerankerUnavailable) as error:
        service(StrategyRepository([chunk(1)], []), FixedReranker(failure=private)).retrieve(
            "query", strategy="hybrid_rerank"
        )
    assert "private model path" not in str(error.value)


def test_semantic_remains_default_and_never_invokes_keyword_or_reranker():
    class SemanticOnlyRepository(StrategyRepository):
        def keyword_search(self, *args):
            raise AssertionError("Default semantic retrieval must not invoke keyword search")
    reranker = FixedReranker(failure=AssertionError("must not run"))
    result = service(SemanticOnlyRepository([chunk(1)], []), reranker).retrieve("query")
    assert [item.chunk_id for item in result.chunks] == [1]
    assert result.strategy == "semantic"
    assert reranker.calls == []


def test_local_cross_encoder_is_lazy_pinned_and_validates_scores(monkeypatch):
    class FakeModel:
        def __init__(self): self.calls = []
        def predict(self, pairs, **kwargs):
            self.calls.append((pairs, kwargs))
            return np.array([.2, .8], dtype=np.float32)
    fake = FakeModel()
    provider = LocalCrossEncoderReranker("model", "b" * 40, "cache", True, 4, 1)
    monkeypatch.setattr(provider, "_load", lambda: fake)
    assert provider._model is None
    scores = provider.score("question", ["one", "two"])
    assert scores == pytest.approx([.2, .8])
    assert provider._model is fake and len(fake.calls) == 1
    assert fake.calls[0][1]["batch_size"] == 4


def test_empty_passages_do_not_load_cross_encoder(monkeypatch):
    provider = LocalCrossEncoderReranker("model", "b" * 40, "cache", True, 4, 1)
    monkeypatch.setattr(provider, "_load", lambda: (_ for _ in ()).throw(AssertionError("must stay lazy")))
    assert provider.score("question", []) == []


@pytest.mark.reranker
@pytest.mark.skipif(os.getenv("RUN_RERANKER_TESTS") != "1",
                    reason="Set RUN_RERANKER_TESTS=1 for the genuine cached cross-encoder")
def test_genuine_cross_encoder_scores_relevant_passage_higher():
    provider = create_reranker_provider(Settings(
        _env_file=None, database_url=None, reranker_local_files_only=True,
    ))
    scores = provider.score(
        "What invalidates the salinity probe calibration?",
        ["The calibration is rejected when drift exceeds 0.4 practical salinity units.",
         "Seedlings are protected by insulated thermal shutters."],
    )
    assert provider.revision == "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    assert scores[0] > scores[1]
