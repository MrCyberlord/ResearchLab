from sqlalchemy import CheckConstraint, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from app.infrastructure.database import Base


class EmbeddingCorpusRow(Base):
    __tablename__ = "embedding_corpus"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_embedding_corpus_singleton"),
        UniqueConstraint("profile", name="uq_embedding_corpus_profile"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    profile: Mapped[str] = mapped_column(String(64), nullable=False)
