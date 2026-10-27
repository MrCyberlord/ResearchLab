"""Explicit offline corpus conversion. No document/text deletion or HTTP endpoint."""
import argparse
import logging

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.application.ingestion import validate_vectors
from app.application.ports import EmbeddingProvider
from app.infrastructure.config import Settings
from app.infrastructure.database import create_session_factory
from app.infrastructure.document_models import ChunkRow, DocumentRow
from app.infrastructure.embedding_corpus import EmbeddingCorpusRow
from app.infrastructure.embeddings import create_embedding_provider


def reembed_corpus(session: Session, provider: EmbeddingProvider) -> int:
    """Commit all documents together, or roll back every changed vector/identity."""
    converted = 0
    with session.begin():
        # Blocks writes (including concurrent ingestion), while reads remain available.
        session.execute(text("LOCK TABLE embedding_corpus, documents, chunks IN SHARE ROW EXCLUSIVE MODE"))
        session.execute(text("SET CONSTRAINTS fk_documents_embedding_profile DEFERRED"))
        for document in session.scalars(select(DocumentRow).order_by(DocumentRow.id)):
            if document.embedding_profile == provider.profile:
                continue
            chunks = list(session.scalars(select(ChunkRow).where(ChunkRow.document_id == document.id)
                                           .order_by(ChunkRow.chunk_order)))
            if len(chunks) != document.chunk_count:
                raise ValueError("Stored chunk count does not match document metadata")
            vectors = provider.embed_documents([chunk.text for chunk in chunks])
            validate_vectors(vectors, len(chunks), provider.dimension)
            for chunk, vector in zip(chunks, vectors, strict=True):
                chunk.embedding = vector
            document.embedding_model = provider.model
            document.embedding_revision = provider.revision
            document.embedding_profile = provider.profile
            document.embedding_dimension = provider.dimension
            session.flush()
            converted += 1
        session.execute(insert(EmbeddingCorpusRow).values(id=1, profile=provider.profile)
                        .on_conflict_do_update(index_elements=["id"], set_={"profile": provider.profile}))
    return converted


def main():
    parser = argparse.ArgumentParser(description="Atomically re-embed stored chunks with the configured provider")
    parser.add_argument("--apply", action="store_true", help="Required explicit write intent")
    args = parser.parse_args()
    if not args.apply:
        parser.error("Pass --apply to re-embed the corpus; this locks ingestion writes until completion")
    settings = Settings()
    engine, factory = create_session_factory(settings)
    try:
        provider = create_embedding_provider(settings)
        provider.embed_query("Check model readiness before acquiring database write locks.")
        with factory() as session:
            count = reembed_corpus(session, provider)
        print(f"Re-embedded {count} documents; IDs, text, page metadata and filenames preserved.")
    except Exception as exc:
        logging.getLogger("researchlab").error("corpus_reembedding_failed")
        raise SystemExit(f"Re-embedding failed ({type(exc).__name__}); transaction rolled back.") from None
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
