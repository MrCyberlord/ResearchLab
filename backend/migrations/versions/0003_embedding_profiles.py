"""Versioned embedding identity and single compatible corpus; preserve legacy vectors."""
import hashlib
import json
from alembic import op
import sqlalchemy as sa

revision = "0003_embedding_profiles"
down_revision = "0002_document_ingestion"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("embedding_corpus",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile", sa.String(64), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("profile", name="uq_embedding_corpus_profile"),
        sa.CheckConstraint("id = 1", name="ck_embedding_corpus_singleton"))
    op.add_column("documents", sa.Column("embedding_profile", sa.String(64), nullable=True))
    op.add_column("documents", sa.Column("embedding_revision", sa.String(100), nullable=True))
    # Frozen historical contract, not imported from mutable application code.
    contract = {"model": "local-hash-v1", "revision": "local-hash-v1",
                "pipeline": "nfkc-casefold-sha256-signed-v1", "dimensions": 384,
                "normalization": "l2", "max_tokens": None}
    lexical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    quarantine = hashlib.sha256(b"legacy-unverified-corpus-v1").hexdigest()
    # One SQL operation handles empty, known lexical, and unknown/mixed corpora.
    op.execute(sa.text("""
        INSERT INTO embedding_corpus (id, profile)
        SELECT 1, CASE WHEN bool_and(embedding_model = 'local-hash-v1' AND embedding_dimension = 384)
                       THEN :lexical ELSE :quarantine END
        FROM documents HAVING count(*) > 0
    """).bindparams(lexical=lexical, quarantine=quarantine))
    op.execute(sa.text("""
        UPDATE documents SET embedding_profile = corpus.profile,
            embedding_revision = CASE WHEN corpus.profile = :lexical
                                      THEN 'local-hash-v1' ELSE 'legacy-unverified' END
        FROM embedding_corpus AS corpus
    """).bindparams(lexical=lexical))
    op.alter_column("documents", "embedding_profile", nullable=False)
    op.alter_column("documents", "embedding_revision", nullable=False)
    op.create_foreign_key("fk_documents_embedding_profile", "documents", "embedding_corpus",
                          ["embedding_profile"], ["profile"], deferrable=True, initially="IMMEDIATE")


def downgrade():
    op.drop_constraint("fk_documents_embedding_profile", "documents", type_="foreignkey")
    op.drop_column("documents", "embedding_revision")
    op.drop_column("documents", "embedding_profile")
    op.drop_table("embedding_corpus")
