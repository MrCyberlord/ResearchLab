from contextlib import contextmanager
from dataclasses import fields

from sqlalchemy import func, literal_column, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.domain.documents import Document, IncompatibleEmbeddings
from app.domain.rag import RetrievedChunk, RetrievalUnavailable
from app.infrastructure.document_models import ChunkRow, DocumentRow
from app.infrastructure.embedding_corpus import EmbeddingCorpusRow


class SQLAlchemyRetrievalRepository:
    def __init__(self, session: Session):
        self.session = session

    @contextmanager
    def _read(self):
        try:
            with self.session.begin():
                yield
        except SQLAlchemyError as exc:
            raise RetrievalUnavailable("The document store is unavailable.") from exc

    def _check_profile_locked(self, profile: str) -> None:
        current = self.session.scalar(select(EmbeddingCorpusRow.profile)
                     .where(EmbeddingCorpusRow.id == 1).with_for_update(read=True))
        if current is not None and current != profile:
            raise IncompatibleEmbeddings()

    def check_profile(self, profile: str) -> None:
        with self._read():
            self._check_profile_locked(profile)

    def search(self, vector, profile, top_k, min_similarity, document_ids):
        with self._read():
            self._check_profile_locked(profile)
            if document_ids == []:
                return []
            distance = ChunkRow.embedding.cosine_distance(vector).label("distance")
            statement = (select(ChunkRow, DocumentRow.filename, distance)
                .join(DocumentRow, DocumentRow.id == ChunkRow.document_id)
                .where(DocumentRow.embedding_profile == profile, distance <= 1.0 - min_similarity))
            if document_ids is not None:
                statement = statement.where(DocumentRow.id.in_(document_ids))
            statement = statement.order_by(distance, ChunkRow.document_id, ChunkRow.chunk_order, ChunkRow.id).limit(top_k)
            return [RetrievedChunk(chunk.id, chunk.document_id, filename, chunk.page_number,
                                   chunk.chunk_order, chunk.text, max(0.0, min(2.0, float(score))))
                    for chunk, filename, score in self.session.execute(statement)]

    def keyword_search(self, query, vector, profile, top_k, document_ids):
        with self._read():
            self._check_profile_locked(profile)
            if document_ids == []:
                return []
            language = literal_column("'english'::regconfig")
            tsquery = func.websearch_to_tsquery(language, query)
            rank = func.ts_rank_cd(ChunkRow.search_vector, tsquery).label("keyword_rank")
            distance = ChunkRow.embedding.cosine_distance(vector).label("distance")
            statement = (select(ChunkRow, DocumentRow.filename, distance, rank)
                .join(DocumentRow, DocumentRow.id == ChunkRow.document_id)
                .where(DocumentRow.embedding_profile == profile,
                       ChunkRow.search_vector.op("@@")(tsquery))
                .order_by(rank.desc(), ChunkRow.document_id, ChunkRow.chunk_order, ChunkRow.id)
                .limit(top_k))
            if document_ids is not None:
                statement = statement.where(DocumentRow.id.in_(document_ids))
            return [RetrievedChunk(chunk.id, chunk.document_id, filename, chunk.page_number,
                                   chunk.chunk_order, chunk.text, max(0.0, min(2.0, float(score))))
                    for chunk, filename, score, _ in self.session.execute(statement)]

    def list_documents(self, limit: int, offset: int):
        with self._read():
            total = self.session.scalar(select(func.count()).select_from(DocumentRow))
            rows = self.session.scalars(select(DocumentRow).order_by(DocumentRow.created_at.desc(), DocumentRow.id)
                                        .limit(limit).offset(offset))
            return [Document(**{field.name: getattr(row, field.name) for field in fields(Document)}) for row in rows], total
