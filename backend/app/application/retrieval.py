import logging
from time import perf_counter
from uuid import UUID

from app.application.ingestion import validate_vectors
from app.application.ports import EmbeddingProvider, RetrievalRepository
from app.domain.documents import EmbeddingFailure
from app.domain.rag import QuestionInvalid, RetrievalResult

logger = logging.getLogger("researchlab")
MAX_QUESTION_CHARS = 2000
MAX_QUESTION_BYTES = 4000


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


class RetrievalService:
    def __init__(self, repository: RetrievalRepository, embeddings: EmbeddingProvider,
                 default_top_k: int = 5, min_similarity: float = 0.25, max_top_k: int = 20):
        self.repository = repository
        self.embeddings = embeddings
        self.default_top_k = default_top_k
        self.min_similarity = min_similarity
        self.max_top_k = max_top_k

    def retrieve(self, query: str, top_k: int | None = None,
                 document_ids: list[UUID] | None = None) -> RetrievalResult:
        query = validate_question(query)
        top_k = self.default_top_k if top_k is None else top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= self.max_top_k:
            raise QuestionInvalid(f"top_k must be between 1 and {self.max_top_k}.")
        if document_ids is not None:
            if len(document_ids) > 100 or any(not isinstance(value, UUID) for value in document_ids):
                raise QuestionInvalid("Select at most 100 valid document IDs.")
            document_ids = list(dict.fromkeys(document_ids))
        started = perf_counter()
        if document_ids == []:
            chunks = []  # Explicitly selecting none must never search all documents.
        else:
            self.repository.check_profile(self.embeddings.profile)
            try:
                vector = self.embeddings.embed_query(query)
                validate_vectors([vector], 1, self.embeddings.dimension)
            except Exception as exc:
                raise EmbeddingFailure("Query embedding generation failed; check the local model cache.") from exc
            # Adapter rechecks/locks profile in the same read transaction as search.
            chunks = self.repository.search(vector, self.embeddings.profile, top_k,
                                            self.min_similarity, document_ids)
        elapsed = round((perf_counter() - started) * 1000, 2)
        logger.info("retrieval_completed", extra={"retrieval_ms": elapsed,
                    "retrieved_chunk_count": len(chunks), "outcome": "ok" if chunks else "empty"})
        return RetrievalResult(chunks, self.embeddings.profile, elapsed)
