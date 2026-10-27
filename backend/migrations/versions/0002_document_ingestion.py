"""Document ingestion and fixed-width pgvector storage."""
from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa

revision = "0002_document_ingestion"
down_revision = "0001_enable_vector"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("embedding_model", sa.String(100), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("chunk_size", sa.Integer(), nullable=False),
        sa.Column("chunk_overlap", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("content_hash", name="uq_documents_content_hash"),
        sa.CheckConstraint("byte_size > 0 AND page_count > 0 AND chunk_count > 0", name="ck_documents_positive_counts"),
        sa.CheckConstraint("embedding_dimension = 384", name="ck_documents_embedding_dimension"),
        sa.CheckConstraint("chunk_size > 0 AND chunk_overlap >= 0 AND chunk_overlap < chunk_size", name="ck_documents_chunk_config"),
        sa.CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="ck_documents_content_hash"),
    )
    op.create_index("ix_documents_created_at", "documents", ["created_at"])
    op.create_table("chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("chunk_order", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("document_id", "chunk_order", name="uq_chunks_document_order"),
        sa.CheckConstraint("page_number > 0 AND chunk_order >= 0", name="ck_chunks_metadata"),
        sa.CheckConstraint("length(text) > 0", name="ck_chunks_nonempty_text"),
    )
    op.create_index("ix_chunks_document_page", "chunks", ["document_id", "page_number"])
    # ANN indexes are deliberately deferred until retrieval and its distance metric exist.


def downgrade():
    op.drop_table("chunks")
    op.drop_table("documents")
