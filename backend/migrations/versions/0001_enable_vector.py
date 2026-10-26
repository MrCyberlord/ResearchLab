"""Enable pgvector; feature tables belong to later milestones."""
from alembic import op

revision = "0001_enable_vector"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade():
    # Preserve a potentially shared extension and any user vector data.
    pass
