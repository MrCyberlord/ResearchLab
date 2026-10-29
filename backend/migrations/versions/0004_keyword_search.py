"""Add generated full-text search vector and GIN index."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004_keyword_search"
down_revision = "0003_embedding_profiles"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("chunks", sa.Column(
        "search_vector", postgresql.TSVECTOR(),
        sa.Computed("to_tsvector('english'::regconfig, text)", persisted=True),
        nullable=False,
    ))
    op.create_index("ix_chunks_search_vector", "chunks", ["search_vector"],
                    unique=False, postgresql_using="gin")


def downgrade():
    op.drop_index("ix_chunks_search_vector", table_name="chunks")
    op.drop_column("chunks", "search_vector")
