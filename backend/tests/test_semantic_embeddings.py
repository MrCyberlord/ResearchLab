"""Real local model tests: relative ranking, not brittle exact similarity scores."""
import math
import os
import pytest
from app.infrastructure.config import Settings
from app.infrastructure.embeddings import MiniLMEmbeddings, create_embedding_provider
from app.domain.embeddings import MINILM_MODEL, MINILM_REVISION, LEXICAL_PROFILE


def test_profile_tracks_revision_and_separates_lexical_vectors():
    first = MiniLMEmbeddings(MINILM_MODEL, MINILM_REVISION, "unused", True, 16, 2)
    changed = MiniLMEmbeddings(MINILM_MODEL, "a" * 40, "unused", True, 16, 2)
    assert first.profile != changed.profile
    assert first.profile != LEXICAL_PROFILE


@pytest.fixture(scope="module")
def semantic_provider():
    if os.getenv("RUN_SEMANTIC_TESTS") != "1":
        pytest.skip("Set RUN_SEMANTIC_TESTS=1 to run real MiniLM inference")
    return create_embedding_provider(Settings(embedding_provider="sentence_transformers", embedding_model=MINILM_MODEL))


@pytest.mark.semantic
@pytest.mark.parametrize("query,relevant,irrelevant", [
    ("How do I regain access after forgetting my password?",
     "Use the account recovery email link to reset a forgotten login password.",
     ["A telescope can observe distant galaxies.", "Bake the bread until its crust is golden."]),
    ("Why is exercise good for the heart?",
     "Regular physical activity strengthens cardiovascular health and lowers blood pressure.",
     ["The database stores transaction records on disk.", "The museum displays medieval paintings."]),
    ("Where can I save records so they survive restarting my computer?",
     "PostgreSQL persists committed data on disk, keeping saved records available after a reboot.",
     ["Fresh vegetables are rich in vitamins.", "Whales migrate through the oceans each year."]),
])
def test_paraphrased_query_prefers_relevant_passage(semantic_provider, query, relevant, irrelevant):
    passages = semantic_provider.embed_documents([relevant, *irrelevant])
    vector = semantic_provider.embed_query(query)
    scores = [sum(a*b for a,b in zip(vector, passage)) for passage in passages]
    assert scores[0] > max(scores[1:]) + 0.1
    assert len(vector) == 384
    assert math.isclose(sum(x*x for x in vector), 1.0, rel_tol=1e-5)


@pytest.mark.semantic
def test_document_query_preprocessing_is_identical(semantic_provider):
    document = semantic_provider.embed_documents(["A research document"])[0]
    query = semantic_provider.embed_query("  \uff21 research\n document  ")
    assert all(math.isclose(a, b, abs_tol=1e-6) for a,b in zip(document, query))
