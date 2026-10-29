from __future__ import annotations

import math
from collections.abc import Iterable

from app.evaluation.dataset import EvaluationReference


def percentile(values: Iterable[float], percentile_value: int) -> float | None:
    """Nearest-rank percentile, deterministic for any nonempty finite sample."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = max(1, math.ceil((percentile_value / 100) * len(ordered)))
    return round(ordered[rank - 1], 2)


def retrieval_metrics(cases: list[dict], ks: list[int]) -> dict:
    """Macro metrics over answerable questions; unanswerable cases have no relevance denominator."""
    answerable = [case for case in cases if case["relevant"]]
    metrics = {}
    for k in ks:
        recalls, precisions, reciprocal_ranks, hits = [], [], [], []
        for case in answerable:
            relevant = set(case["relevant"])
            retrieved = case["retrieved"][:k]
            matched = sum(reference in relevant for reference in retrieved)
            recalls.append(matched / len(relevant))
            precisions.append(matched / k)
            first = next((rank for rank, reference in enumerate(retrieved, 1) if reference in relevant), None)
            reciprocal_ranks.append(0.0 if first is None else 1.0 / first)
            hits.append(1.0 if first is not None else 0.0)
        denominator = len(answerable)
        mean = lambda values: round(sum(values) / denominator, 6) if denominator else 0.0
        metrics[str(k)] = {
            "recall_at_k": mean(recalls), "precision_at_k": mean(precisions),
            "mrr_at_k": mean(reciprocal_ranks), "hit_rate_at_k": mean(hits),
            "answerable_question_count": denominator,
        }
    return metrics


def answer_metrics(cases: list[dict]) -> dict:
    """Dataset-reference and quote-presence checks only; this does not measure entailment."""
    completed = [case for case in cases if case.get("status") != "error"]
    answerable = [case for case in completed if case["relevant"]]
    unanswerable = [case for case in completed if not case["relevant"]]
    citations = [citation for case in completed for citation in case.get("citations", [])]
    relevant_citations = sum(citation["reference"] in set(case["relevant"])
                             for case in completed for citation in case.get("citations", []))
    expected = sum(len(set(case["relevant"])) for case in answerable)
    covered = sum(len(set(case["relevant"]) & {citation["reference"] for citation in case.get("citations", [])})
                  for case in answerable)
    correct_behavior = sum((case["status"] == "insufficient_evidence") == (not case["relevant"])
                           for case in completed)
    return {
        "citation_precision": round(relevant_citations / len(citations), 6) if citations else None,
        "citation_coverage": round(covered / expected, 6) if expected else None,
        "mechanical_quote_validity": round(sum(item["quote_valid"] for item in citations) / len(citations), 6)
                                     if citations else None,
        "abstention_behavior_accuracy": round(correct_behavior / len(completed), 6) if completed else None,
        "unanswerable_abstention_rate": round(sum(case["status"] == "insufficient_evidence" for case in unanswerable)
                                               / len(unanswerable), 6) if unanswerable else None,
        "answerable_response_rate": round(sum(case["status"] == "answered" for case in answerable)
                                           / len(answerable), 6) if answerable else None,
        "completed_question_count": len(completed), "citation_count": len(citations),
        "expected_reference_count": expected,
        "factual_entailment_measured": False,
    }
