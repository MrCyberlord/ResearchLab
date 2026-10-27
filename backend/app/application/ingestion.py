import hashlib
import math
from pathlib import PurePosixPath
from uuid import uuid4

from app.application.ports import DocumentRepository, EmbeddingProvider, PDFExtractor
from app.domain.documents import (
    Chunk, Document, DuplicateDocument, EmbeddingFailure, InvalidPDF, PageText, PDFLimitExceeded,
)


def split_pages(pages: list[PageText], document_id, size: int, overlap: int, max_chunks: int = 5000) -> list[Chunk]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Chunk overlap must be smaller than the positive chunk size")
    chunks = []
    for page in pages:
        text = page.text.replace("\x00", "").strip()
        for start in range(0, len(text), size - overlap):
            piece = text[start:start + size]
            if piece.strip():
                if len(chunks) >= max_chunks:
                    raise PDFLimitExceeded("PDF exceeds the configured chunk limit.")
                chunks.append(Chunk(document_id, page.number, len(chunks), piece))
            if start + size >= len(text):
                break
    return chunks


def validate_vectors(vectors: list[list[float]], count: int, dimension: int) -> None:
    if dimension != 384 or len(vectors) != count or any(
        len(vector) != dimension or not all(math.isfinite(value) for value in vector)
        or not math.isclose(sum(value * value for value in vector), 1.0, rel_tol=1e-4, abs_tol=1e-4)
        for vector in vectors
    ):
        raise ValueError("Expected finite, normalized 384-dimensional embeddings")


class IngestionService:
    def __init__(self, repository: DocumentRepository, extractor: PDFExtractor,
                 embeddings: EmbeddingProvider, chunk_size: int, chunk_overlap: int, max_chunks: int = 5000):
        self.repository = repository
        self.extractor = extractor
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks

    def ingest(self, data: bytes, filename: str) -> Document:
        digest = hashlib.sha256(data).hexdigest()
        # A short read transaction avoids keeping a connection during extraction.
        with self.repository.transaction():
            self.repository.check_profile(self.embeddings.profile)
            existing = self.repository.find_by_hash(digest)
            if existing:
                raise DuplicateDocument(existing.id)
        pages = self.extractor.extract(data)
        document_id = uuid4()
        chunks = split_pages(pages, document_id, self.chunk_size, self.chunk_overlap, self.max_chunks)
        if not chunks:
            raise InvalidPDF("PDF has no extractable text. Scanned PDFs require OCR, which is not supported.")
        try:
            vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
            validate_vectors(vectors, len(chunks), self.embeddings.dimension)
        except Exception as exc:
            raise EmbeddingFailure("Local embedding generation failed; no document was saved.") from exc
        # A supplied filename is metadata only, never a filesystem destination.
        safe_name = PurePosixPath(filename.replace("\\", "/")).name
        safe_name = "".join(c for c in safe_name if c.isprintable())[:255] or "document.pdf"
        document = Document(document_id, safe_name, digest, len(data), len(pages), len(chunks),
                            self.embeddings.model, self.embeddings.dimension,
                            self.chunk_size, self.chunk_overlap, self.embeddings.profile, self.embeddings.revision)
        with self.repository.transaction():
            self.repository.bind_profile(self.embeddings.profile)
            # Recheck after the slow work; the unique constraint closes the race.
            existing = self.repository.find_by_hash(digest)
            if existing:
                raise DuplicateDocument(existing.id)
            self.repository.add(document, chunks, vectors)
        return document
