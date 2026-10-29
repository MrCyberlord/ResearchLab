import json
from pathlib import Path
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest

from app.domain.rag import RetrievalResult, RetrievedChunk
from app.evaluation.dataset import EvaluationReference, default_dataset_path, load_dataset
from app.evaluation.metrics import answer_metrics, percentile, retrieval_metrics
from app.evaluation.runner import EvaluationEnvironment, EvaluationRunner, compare_results


def ref(document="doc", page=1, chunk_order=0):
    return EvaluationReference(document, page, chunk_order)


def test_retrieval_metrics_multiple_relevant_and_empty_results():
    cases = [
        {"relevant": [ref("a")], "retrieved": [ref("x"), ref("a")]},
        {"relevant": [ref("c"), ref("d")], "retrieved": [ref("c"), ref("d")]},
        {"relevant": [], "retrieved": [ref("x")]},  # No relevance denominator.
    ]
    result = retrieval_metrics(cases, [1, 2])
    assert result["1"] == {
        "recall_at_k": 0.25, "precision_at_k": 0.5, "mrr_at_k": 0.5,
        "hit_rate_at_k": 0.5, "answerable_question_count": 2,
    }
    assert result["2"]["recall_at_k"] == 1
    assert result["2"]["precision_at_k"] == .75
    assert result["2"]["mrr_at_k"] == .75
    assert result["2"]["hit_rate_at_k"] == 1
    assert retrieval_metrics([{"relevant": [ref()], "retrieved": []}], [1])["1"]["recall_at_k"] == 0


def test_answer_metrics_are_mechanical_and_score_abstention():
    cases = [
        {"relevant": [ref("a"), ref("b")], "status": "answered",
         "citations": [{"reference": ref("a"), "quote_valid": True}]},
        {"relevant": [], "status": "answered",
         "citations": [{"reference": ref("x"), "quote_valid": False}]},
        {"relevant": [], "status": "error", "citations": []},
    ]
    result = answer_metrics(cases)
    assert result["citation_precision"] == .5
    assert result["citation_coverage"] == .5
    assert result["mechanical_quote_validity"] == .5
    assert result["abstention_behavior_accuracy"] == .5
    assert result["unanswerable_abstention_rate"] == 0
    assert result["answerable_response_rate"] == 1
    assert result["completed_question_count"] == 2
    assert result["factual_entailment_measured"] is False


def test_nearest_rank_latency_percentiles_and_empty_sample():
    assert percentile([], 95) is None
    assert percentile([40, 10, 30, 20], 50) == 20
    assert percentile([40, 10, 30, 20], 95) == 40


def test_versioned_dataset_is_licensed_nontrivial_and_internally_consistent():
    dataset = load_dataset(default_dataset_path())
    assert dataset.version == "researchlab-rag-eval-v1"
    assert dataset.license == "CC0-1.0"
    assert len(dataset.documents) == 4 and len(dataset.questions) == 8
    assert any(len(question.relevant) > 1 for question in dataset.questions)
    assert any(question.document_filter for question in dataset.questions)
    assert sum(question.expect_abstain for question in dataset.questions) == 2
    assert (default_dataset_path().parent / "LICENSE.md").is_file()
    baseline = default_dataset_path().parents[2] / "baselines" / "minilm-v1.json"
    assert json.loads(baseline.read_text("utf-8"))["dataset"]["sha256"] == dataset.checksum


def test_baseline_comparison_detects_regression_and_incompatible_configuration():
    base = {
        "dataset": {"sha256": "same"}, "embedding": {"profile": "same"},
        "retrieval": {"configuration": {"k_values": [1]}, "metrics": {
            "1": {"recall_at_k": 1.0, "precision_at_k": .5, "mrr_at_k": 1.0, "hit_rate_at_k": 1.0}
        }},
    }
    current = {
        "dataset": {"sha256": "same"}, "embedding": {"profile": "same"},
        "retrieval": {"configuration": {"k_values": [1]}, "metrics": {
            "1": {"recall_at_k": .5, "precision_at_k": .5, "mrr_at_k": 1.0, "hit_rate_at_k": 1.0}
        }},
    }
    comparison = compare_results(current, base)
    assert comparison["compatible"] is True and comparison["regression"] is True
    current["embedding"]["profile"] = "changed"
    assert compare_results(current, base) == {
        "compatible": False, "retrieval_metric_deltas": {}, "regression": False,
    }


class RecordingRetrieval:
    def __init__(self, dataset):
        self.calls = []
        self.embeddings = SimpleNamespace(model="test", revision="r1", dimension=384, profile="p1")
        self.dataset = dataset

    def retrieve(self, query, top_k, document_ids):
        self.calls.append((query, top_k, document_ids))
        question = next(item for item in self.dataset.questions if item.question == query)
        chunks = []
        for reference in question.relevant[:top_k]:
            document_id = uuid5(NAMESPACE_URL, reference.document)
            chunks.append(RetrievedChunk(len(chunks) + 1, document_id, reference.document + ".txt",
                                         reference.page, reference.chunk_order, "synthetic evidence", .1))
        return RetrievalResult(chunks, "p1", 1.25)


def _fake_run(tmp_path: Path):
    dataset = load_dataset(default_dataset_path())
    ids = {document.key: uuid5(NAMESPACE_URL, document.key) for document in dataset.documents}
    retrieval = RecordingRetrieval(dataset)
    environment = EvaluationEnvironment(retrieval, None, ids, {value: key for key, value in ids.items()}, "test")
    settings = SimpleNamespace(embedding_provider="test")
    result = EvaluationRunner(dataset, environment, settings, "fixed", tmp_path).run([3, 1], 3)
    return dataset, retrieval, result


def test_runner_is_reproducible_and_applies_document_filters(tmp_path):
    dataset, first_retrieval, first = _fake_run(tmp_path)
    _, second_retrieval, second = _fake_run(tmp_path)
    assert first["retrieval"] == second["retrieval"]
    assert first["retrieval"]["configuration"]["k_values"] == [1, 3]
    filtered_index = next(index for index, question in enumerate(dataset.questions) if question.document_filter)
    expected_ids = [uuid5(NAMESPACE_URL, key) for key in dataset.questions[filtered_index].document_filter]
    assert first_retrieval.calls[filtered_index][2] == expected_ids
    assert second_retrieval.calls == first_retrieval.calls
    assert first["answer_evaluation"]["status"] == "disabled"


@pytest.mark.parametrize("ks", [[], [0], [21], [True]])
def test_runner_rejects_invalid_k_values(tmp_path, ks):
    dataset = load_dataset(default_dataset_path())
    ids = {document.key: uuid5(NAMESPACE_URL, document.key) for document in dataset.documents}
    retrieval = RecordingRetrieval(dataset)
    environment = EvaluationEnvironment(retrieval, None, ids, {value: key for key, value in ids.items()}, "test")
    with pytest.raises(ValueError):
        EvaluationRunner(dataset, environment, SimpleNamespace(embedding_provider="test"),
                         "fixed", tmp_path).run(ks, 3)
