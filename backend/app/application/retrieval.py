import logging
import math
from time import perf_counter
from uuid import UUID

from app.application.ingestion import validate_vectors
from app.application.ports import EmbeddingProvider, RerankerProvider, RetrievalRepository
from app.domain.documents import EmbeddingFailure
from app.domain.rag import (QuestionInvalid, RerankerUnavailable, RetrievalResult,
                            RetrievalStrategy, RetrievedChunk)

logger = logging.getLogger("researchlab")
MAX_QUESTION_CHARS = 2000
MAX_QUESTION_BYTES = 4000
STRATEGIES = {"semantic", "hybrid", "hybrid_rerank"}


def validate_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise QuestionInvalid("Enter a nonempty question.")
    question = question.strip()
    try:
        encoded = question.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise QuestionInvalid("Question contains invalid Unicode.") from exc
    if len(question) > MAX_QUESTION_CHARS or len(encoded) > MAX_QUESTION_BYTES:
        raise QuestionInvalid("Question exceeds the character or UTF-8 byte limit.")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in question):
        raise QuestionInvalid("Question contains unsupported control characters.")
    return question


def reciprocal_rank_fusion(rankings: list[list[RetrievedChunk]], constant: int,
                           limit: int) -> list[RetrievedChunk]:
    """Fuse rankings by stable database chunk ID; duplicate IDs contribute once per ranking."""
    scores: dict[int, float] = {}
    chunks: dict[int, RetrievedChunk] = {}
    best_rank: dict[int, int] = {}
    for ranking in rankings:
        seen = set()
        for rank, chunk in enumerate(ranking, 1):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks.setdefault(chunk.chunk_id, chunk)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (constant + rank)
            best_rank[chunk.chunk_id] = min(rank, best_rank.get(chunk.chunk_id, rank))
    ordered = sorted(chunks.values(), key=lambda chunk: (
        -scores[chunk.chunk_id], best_rank[chunk.chunk_id], str(chunk.document_id),
        chunk.chunk_order, chunk.chunk_id,
    ))
    return ordered[:limit]


class RetrievalService:
    def __init__(self, repository: RetrievalRepository, embeddings: EmbeddingProvider,
                 default_top_k: int = 5, min_similarity: float = 0.25, max_top_k: int = 20,
                 default_strategy: RetrievalStrategy = "semantic", semantic_candidates: int = 20,
                 keyword_candidates: int = 20, rrf_constant: int = 60,
                 reranker: RerankerProvider | None = None, reranker_candidates: int = 20):
        self.repository = repository
        self.embeddings = embeddings
        self.default_top_k = default_top_k
        self.min_similarity = min_similarity
        self.max_top_k = max_top_k
        self.default_strategy = default_strategy
        self.semantic_candidates = semantic_candidates
        self.keyword_candidates = keyword_candidates
        self.rrf_constant = rrf_constant
        self.reranker = reranker
        self.reranker_candidates = reranker_candidates

    def retrieve(self, query: str, top_k: int | None = None,
                 document_ids: list[UUID] | None = None,
                 strategy: RetrievalStrategy | None = None) -> RetrievalResult:
        query = validate_question(query)
        top_k = self.default_top_k if top_k is None else top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= self.max_top_k:
            raise QuestionInvalid(f"top_k must be between 1 and {self.max_top_k}.")
        if document_ids is not None:
            if len(document_ids) > 100 or any(not isinstance(value, UUID) for value in document_ids):
                raise QuestionInvalid("Select at most 100 valid document IDs.")
            document_ids = list(dict.fromkeys(document_ids))
        strategy = self.default_strategy if strategy is None else strategy
        if strategy not in STRATEGIES:
            raise QuestionInvalid("strategy must be semantic, hybrid or hybrid_rerank.")
        started = perf_counter()
        semantic_count = keyword_count = 0
        rerank_ms = 0.0
        if document_ids == []:
            chunks = []  # Explicitly selecting none must never search all documents.
        else:
            self.repository.check_profile(self.embeddings.profile)
            try:
                vector = self.embeddings.embed_query(query)
                validate_vectors([vector], 1, self.embeddings.dimension)
            except Exception as exc:
                raise EmbeddingFailure("Query embedding generation failed; check the local model cache.") from exc
            # Each adapter call rechecks/locks the profile in its own read transaction.
            if strategy == "semantic":
                chunks = self.repository.search(vector, self.embeddings.profile, top_k,
                                                self.min_similarity, document_ids)
                semantic_count = len(chunks)
            else:
                semantic = self.repository.search(
                    vector, self.embeddings.profile, self.semantic_candidates,
                    self.min_similarity, document_ids,
                )
                keyword = self.repository.keyword_search(
                    query, vector, self.embeddings.profile, self.keyword_candidates, document_ids,
                )
                semantic_count, keyword_count = len(semantic), len(keyword)
                fused_limit = self.reranker_candidates if strategy == "hybrid_rerank" else top_k
                fused = reciprocal_rank_fusion([semantic, keyword], self.rrf_constant, fused_limit)
                if strategy == "hybrid":
                    chunks = fused[:top_k]
                else:
                    if self.reranker is None:
                        raise RerankerUnavailable("The local reranker is not configured.")
                    rerank_started = perf_counter()
                    try:
                        scores = self.reranker.score(query, [chunk.text for chunk in fused])
                        if len(scores) != len(fused) or any(not math.isfinite(score) for score in scores):
                            raise ValueError("Invalid reranker scores")
                    except Exception as exc:
                        raise RerankerUnavailable(
                            "Local reranking failed; check the pinned model cache and runtime."
                        ) from exc
                    rerank_ms = round((perf_counter() - rerank_started) * 1000, 2)
                    chunks = [chunk for _, chunk in sorted(
                        enumerate(fused), key=lambda item: (-scores[item[0]], item[0])
                    )[:top_k]]
        elapsed = round((perf_counter() - started) * 1000, 2)
        logger.info("retrieval_completed", extra={"retrieval_ms": elapsed,
                    "rerank_ms": rerank_ms, "retrieval_strategy": strategy,
                    "semantic_candidate_count": semantic_count, "keyword_candidate_count": keyword_count,
                    "retrieved_chunk_count": len(chunks), "outcome": "ok" if chunks else "empty"})
        return RetrievalResult(chunks, self.embeddings.profile, elapsed, strategy)
