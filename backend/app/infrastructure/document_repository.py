from contextlib import contextmanager
from dataclasses import asdict

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.domain.documents import Chunk, Document, DuplicateDocument, IncompatibleEmbeddings
from app.infrastructure.document_models import ChunkRow, DocumentRow
from app.infrastructure.embedding_corpus import EmbeddingCorpusRow


class SQLAlchemyDocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    @contextmanager
    def transaction(self):
        try:
            with self.session.begin():
                yield
        except IntegrityError as exc:
            # Match only the deduplication constraint; other integrity errors must surface.
            if getattr(getattr(exc.orig, "diag", None), "constraint_name", None) == "uq_documents_content_hash":
                raise DuplicateDocument() from exc
            raise

    def check_profile(self, profile: str) -> None:
        current = self.session.scalar(select(EmbeddingCorpusRow.profile).where(EmbeddingCorpusRow.id == 1))
        if current is not None and current != profile:
            raise IncompatibleEmbeddings()

    def bind_profile(self, profile: str) -> None:
        # Concurrent first writers serialize on the singleton primary key.
        self.session.execute(insert(EmbeddingCorpusRow).values(id=1, profile=profile).on_conflict_do_nothing())
        self.check_profile(profile)

    def find_by_hash(self, content_hash: str) -> Document | None:
        row = self.session.scalar(select(DocumentRow).where(DocumentRow.content_hash == content_hash))
        if row is None:
            return None
        return Document(**{field: getattr(row, field) for field in Document.__dataclass_fields__})

    def add(self, document: Document, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.session.add(DocumentRow(**asdict(document)))
        self.session.flush()  # Parent must exist before bulk chunk insertion.
        self.session.add_all([ChunkRow(**asdict(chunk), embedding=vector)
                              for chunk, vector in zip(chunks, embeddings, strict=True)])
        self.session.flush()
