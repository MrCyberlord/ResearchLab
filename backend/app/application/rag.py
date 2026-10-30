"""RAG application service composed from retrieval and shared grounding."""
from uuid import UUID

from app.application.grounding import GroundedAnswerService
from app.application.ports import LLMProvider
from app.application.retrieval import RetrievalService, validate_question
from app.domain.rag import RagResult, RetrievalStrategy


class RagService:
    def __init__(self, retrieval: RetrievalService, llm: LLMProvider,
                 context_max_bytes: int = 4000, prompt_max_bytes: int = 6000):
        self.retrieval = retrieval
        self.llm = llm
        self.grounding = GroundedAnswerService(llm, context_max_bytes, prompt_max_bytes)

    def answer(self, question: str, top_k: int | None = None,
               document_ids: list[UUID] | None = None,
               strategy: RetrievalStrategy | None = None) -> RagResult:
        question = validate_question(question)
        retrieved = self.retrieval.retrieve(question, top_k, document_ids, strategy)
        grounded = self.grounding.answer(
            question, retrieved.chunks, retrieval_ms=retrieved.retrieval_ms,
        )
        return RagResult(
            status=grounded.status, answer=grounded.answer, message=grounded.message,
            citations=grounded.citations, retrieval_ms=retrieved.retrieval_ms,
            generation_ms=grounded.generation_ms,
            retrieved_chunk_count=len(retrieved.chunks),
            context_chunk_count=grounded.context_chunk_count,
            embedding_profile=retrieved.embedding_profile,
            llm_provider=self.llm.provider, llm_model=self.llm.model,
            usage=grounded.usage, retrieval_strategy=retrieved.strategy,
        )
