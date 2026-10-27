from dataclasses import dataclass
from uuid import UUID
from app.domain.embeddings import LEXICAL_PROFILE


@dataclass(frozen=True)
class PageText:
    number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    document_id: UUID
    page_number: int
    chunk_order: int  # Zero-based, global within the document.
    text: str


@dataclass(frozen=True)
class Document:
    id: UUID
    filename: str
    content_hash: str
    byte_size: int
    page_count: int
    chunk_count: int
    embedding_model: str
    embedding_dimension: int
    chunk_size: int
    chunk_overlap: int
    embedding_profile: str = LEXICAL_PROFILE
    embedding_revision: str = "local-hash-v1"


class IngestionError(Exception):
    """A safe, client-facing ingestion failure."""


class InvalidPDF(IngestionError):
    pass


class PDFLimitExceeded(IngestionError):
    pass


class DuplicateDocument(IngestionError):
    def __init__(self, document_id: UUID | None = None):
        self.document_id = document_id
        super().__init__("This exact PDF has already been ingested.")


class EmbeddingFailure(IngestionError):
    pass


class IncompatibleEmbeddings(IngestionError):
    def __init__(self):
        super().__init__("The database contains a different embedding model/profile. Run the explicit corpus re-embedding command before uploading with this configuration; existing data was preserved.")
