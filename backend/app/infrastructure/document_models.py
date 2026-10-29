from datetime import datetime
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, Computed, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database import Base
from app.infrastructure.embedding_corpus import EmbeddingCorpusRow  # Register corpus table.

EMBEDDING_DIMENSION = 384


class DocumentRow(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_documents_content_hash"),
        CheckConstraint("byte_size > 0 AND page_count > 0 AND chunk_count > 0", name="ck_documents_positive_counts"),
        CheckConstraint("embedding_dimension = 384", name="ck_documents_embedding_dimension"),
        CheckConstraint("chunk_size > 0 AND chunk_overlap >= 0 AND chunk_overlap < chunk_size", name="ck_documents_chunk_config"),
        CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="ck_documents_content_hash"),
        Index("ix_documents_created_at", "created_at"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    content_hash: Mapped[str] = mapped_column(String(64))
    byte_size: Mapped[int] = mapped_column(Integer)
    page_count: Mapped[int] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer)
    embedding_model: Mapped[str] = mapped_column(String(100))
    embedding_profile: Mapped[str] = mapped_column(String(64), ForeignKey(
        "embedding_corpus.profile", name="fk_documents_embedding_profile", deferrable=True, initially="IMMEDIATE"))
    embedding_revision: Mapped[str] = mapped_column(String(100))
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    chunk_size: Mapped[int] = mapped_column(Integer)
    chunk_overlap: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChunkRow(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_order", name="uq_chunks_document_order"),
        CheckConstraint("page_number > 0 AND chunk_order >= 0", name="ck_chunks_metadata"),
        CheckConstraint("length(text) > 0", name="ck_chunks_nonempty_text"),
        Index("ix_chunks_document_page", "document_id", "page_number"),
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
    )
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    page_number: Mapped[int] = mapped_column(Integer)
    chunk_order: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english'::regconfig, text)", persisted=True), nullable=False
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSION))
