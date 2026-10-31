"""Validate and map committed evaluation artifacts into safe aggregate summaries."""
from datetime import datetime
import math

from app.application.ports import EvaluationArtifactReader
from app.domain.evaluation_dashboard import (
    EvaluationArtifactAccessDenied, EvaluationArtifactIncomplete,
    EvaluationArtifactIncompatible, EvaluationArtifactInvalid,
    EvaluationArtifactMissing,
)

ARTIFACT_IDS = (
    "rag_baseline", "rag_strategy_reference", "agent_baseline",
    "agent_evidence_observation",
)


def _mapping(value, field: str) -> dict:
    if not isinstance(value, dict):
        raise EvaluationArtifactIncomplete(f"{field} must be an object")
    return value


def _text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationArtifactIncomplete(f"{field} must be non-empty text")
    return value


def _integer(value, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvaluationArtifactIncomplete(f"{field} must be an integer")
    return value


def _number(value, field: str, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationArtifactIncomplete(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise EvaluationArtifactIncomplete(f"{field} must be finite")
    return result


def _ratio(value, field: str) -> float | None:
    if value is None:
        return None
    result = _number(value, field)
    if result > 1:
        raise EvaluationArtifactIncomplete(f"{field} must be a ratio")
    return result


def _boolean(value, field: str) -> bool:
    if not isinstance(value, bool):
        raise EvaluationArtifactIncomplete(f"{field} must be boolean")
    return value


def _timestamp(value, field: str) -> str:
    result = _text(value, field)
    try:
        datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvaluationArtifactIncomplete(f"{field} must be ISO-8601") from exc
    return result


def _schema(data: dict, expected: int, artifact_id: str) -> None:
    if data.get("schema_version") != expected:
        raise EvaluationArtifactIncompatible(
            f"{artifact_id} schema version is unsupported"
        )


def _latency(value, field: str) -> dict:
    data = _mapping(value, field)
    return {
        "sample_count": _integer(data.get("sample_count"), f"{field}.sample_count", 1),
        "p50_ms": _number(data.get("p50"), f"{field}.p50"),
        "p95_ms": _number(data.get("p95"), f"{field}.p95"),
    }


def _distribution(value, field: str) -> dict:
    data = _mapping(value, field)
    return {
        "sample_count": _integer(data.get("sample_count"), f"{field}.sample_count"),
        "mean": _number(data.get("mean"), f"{field}.mean"),
        "minimum": _integer(data.get("min"), f"{field}.min"),
        "maximum": _integer(data.get("max"), f"{field}.max"),
    }


def _run(value, field: str) -> dict:
    data = _mapping(value, field)
    return {
        "run_id": _text(data.get("run_id"), f"{field}.run_id"),
        "created_at_utc": _timestamp(data.get("created_at_utc"), f"{field}.created_at_utc"),
    }


def _rag_metric_rows(metrics: dict, k_values: list[int], answerable: int,
                     field: str) -> list[dict]:
    rows = []
    for k in k_values:
        values = _mapping(metrics.get(str(k)), f"{field}.{k}")
        reported = values.get("answerable_question_count", answerable)
        if _integer(reported, f"{field}.{k}.answerable_question_count", 1) != answerable:
            raise EvaluationArtifactIncompatible("RAG metric denominator does not match dataset")
        rows.append({
            "k": k,
            "recall_at_k": _ratio(values.get("recall_at_k"), f"{field}.{k}.recall_at_k"),
            "precision_at_k": _ratio(values.get("precision_at_k"), f"{field}.{k}.precision_at_k"),
            "mrr_at_k": _ratio(values.get("mrr_at_k"), f"{field}.{k}.mrr_at_k"),
            "hit_rate_at_k": _ratio(values.get("hit_rate_at_k"), f"{field}.{k}.hit_rate_at_k"),
            "answerable_question_count": answerable,
        })
    return rows


def _map_rag_baseline(data: dict) -> dict:
    _schema(data, 1, "rag_baseline")
    dataset = _mapping(data.get("dataset"), "dataset")
    embedding = _mapping(data.get("embedding"), "embedding")
    retrieval = _mapping(data.get("retrieval"), "retrieval")
    configuration = _mapping(retrieval.get("configuration"), "retrieval.configuration")
    k_values_raw = configuration.get("k_values")
    if not isinstance(k_values_raw, list) or not k_values_raw:
        raise EvaluationArtifactIncomplete("retrieval.configuration.k_values is required")
    k_values = [_integer(item, "retrieval.configuration.k_values", 1) for item in k_values_raw]
    if sorted(set(k_values)) != k_values:
        raise EvaluationArtifactIncomplete("retrieval K values must be unique and sorted")
    answerable = _integer(dataset.get("answerable_question_count"), "dataset.answerable_question_count", 1)
    unanswerable = _integer(dataset.get("unanswerable_question_count"), "dataset.unanswerable_question_count")
    question_count = _integer(dataset.get("question_count"), "dataset.question_count", 1)
    if answerable + unanswerable != question_count:
        raise EvaluationArtifactIncompatible("RAG dataset question counts are inconsistent")
    strategy = {
        "strategy": "semantic",
        "reference_only": False,
        "metrics": _rag_metric_rows(
            _mapping(retrieval.get("metrics"), "retrieval.metrics"),
            k_values, answerable, "retrieval.metrics",
        ),
        "latency": _latency(
            retrieval.get("latency_ms_reference_only"),
            "retrieval.latency_ms_reference_only",
        ),
    }
    return {
        "dataset": {
            "version": _text(dataset.get("version"), "dataset.version"),
            "title": _text(dataset.get("title"), "dataset.title"),
            "license": _text(dataset.get("license"), "dataset.license"),
            "document_count": _integer(dataset.get("document_count"), "dataset.document_count", 1),
            "question_count": question_count,
            "answerable_question_count": answerable,
            "unanswerable_question_count": unanswerable,
            "sha256": _text(dataset.get("sha256"), "dataset.sha256"),
        },
        "baseline_run": _run(data.get("run"), "run"),
        "strategy_reference_run": None,
        "model": {
            "provider": _text(embedding.get("provider"), "embedding.provider"),
            "model": _text(embedding.get("model"), "embedding.model"),
            "revision": _text(embedding.get("revision"), "embedding.revision"),
            "dimension": _integer(embedding.get("dimension"), "embedding.dimension", 1),
            "normalization": _text(embedding.get("normalization"), "embedding.normalization"),
        },
        "settings": {
            "default_strategy": "semantic",
            "k_values": k_values,
            "min_similarity": _number(configuration.get("min_similarity"), "retrieval.configuration.min_similarity"),
            "distance": _text(configuration.get("distance"), "retrieval.configuration.distance"),
            "chunk_size": _integer(configuration.get("chunk_size"), "retrieval.configuration.chunk_size", 1),
            "chunk_overlap": _integer(configuration.get("chunk_overlap"), "retrieval.configuration.chunk_overlap"),
            "semantic_candidate_count": None,
            "keyword_candidate_count": None,
            "rrf_fusion_constant": None,
        },
        "strategies": [strategy],
        "metric_definitions": [
            {"key": "recall_at_k", "label": "Recall@K", "definition": "Relevant references retrieved in the top K divided by all relevant references, macro-averaged over answerable questions.", "denominator": f"{answerable} answerable questions"},
            {"key": "precision_at_k", "label": "Precision@K", "definition": "Relevant retrieved references divided by K, macro-averaged over answerable questions.", "denominator": f"{answerable} answerable questions"},
            {"key": "mrr_at_k", "label": "MRR@K", "definition": "Reciprocal rank of the first relevant result in the top K, macro-averaged over answerable questions.", "denominator": f"{answerable} answerable questions"},
            {"key": "hit_rate_at_k", "label": "Hit Rate@K", "definition": "Share of answerable questions with at least one relevant result in the top K.", "denominator": f"{answerable} answerable questions"},
        ],
        "comparability_note": "RAG strategy metrics share one synthetic dataset and relevance judgments. They are not directly comparable to Agentic decision metrics.",
    }


def _attach_rag_reference(summary: dict, data: dict) -> None:
    _schema(data, 2, "rag_strategy_reference")
    dataset = _mapping(data.get("dataset"), "strategy.dataset")
    if (dataset.get("version") != summary["dataset"]["version"]
            or dataset.get("sha256") != summary["dataset"]["sha256"]):
        raise EvaluationArtifactIncompatible("RAG strategy reference uses another dataset")
    if (_integer(dataset.get("answerable_question_count"), "strategy.dataset.answerable_question_count", 1)
            != summary["dataset"]["answerable_question_count"]):
        raise EvaluationArtifactIncompatible("RAG strategy answerable denominator differs")
    configuration = _mapping(data.get("configuration"), "strategy.configuration")
    k_values = summary["settings"]["k_values"]
    if configuration.get("k_values") != k_values:
        raise EvaluationArtifactIncompatible("RAG strategy K values differ")
    strategies = _mapping(data.get("strategies"), "strategies")
    mapped = []
    for name in ("semantic", "hybrid", "hybrid_rerank"):
        values = _mapping(strategies.get(name), f"strategies.{name}")
        mapped.append({
            "strategy": name,
            "reference_only": True,
            "metrics": _rag_metric_rows(
                _mapping(values.get("metrics"), f"strategies.{name}.metrics"),
                k_values, summary["dataset"]["answerable_question_count"],
                f"strategies.{name}.metrics",
            ),
            "latency": _latency(values.get("latency_ms"), f"strategies.{name}.latency_ms"),
        })
    conclusion = _mapping(data.get("conclusion"), "conclusion")
    default_strategy = _text(conclusion.get("default_strategy"), "conclusion.default_strategy")
    if default_strategy not in {item["strategy"] for item in mapped}:
        raise EvaluationArtifactIncomplete("Default RAG strategy is unsupported")
    summary["strategies"] = mapped
    summary["strategy_reference_run"] = _run(data.get("run"), "strategy.run")
    summary["settings"].update({
        "default_strategy": default_strategy,
        "semantic_candidate_count": _integer(configuration.get("semantic_candidate_count"), "configuration.semantic_candidate_count", 1),
        "keyword_candidate_count": _integer(configuration.get("keyword_candidate_count"), "configuration.keyword_candidate_count", 1),
        "rrf_fusion_constant": _integer(configuration.get("rrf_fusion_constant"), "configuration.rrf_fusion_constant", 1),
    })


def _agent_metric(key: str, label: str, metrics: dict, denominator_name: str,
                  denominator: int, definition: str) -> dict:
    return {
        "key": key, "label": label, "value": _ratio(metrics.get(key), f"metrics.{key}"),
        "denominator_name": denominator_name, "denominator": denominator,
        "definition": definition,
    }


def _map_agent_baseline(data: dict) -> dict:
    _schema(data, 1, "agent_baseline")
    dataset = _mapping(data.get("dataset"), "agent.dataset")
    agent = _mapping(data.get("agent"), "agent")
    retrieval = _mapping(data.get("retrieval"), "agent.retrieval")
    metrics = _mapping(data.get("metrics"), "agent.metrics")
    denominators = _mapping(metrics.get("denominators"), "agent.metrics.denominators")
    attempted = _integer(denominators.get("attempted_runs"), "denominators.attempted_runs", 1)
    completed = _integer(denominators.get("completed_runs"), "denominators.completed_runs")
    failed = _integer(metrics.get("error_count"), "metrics.error_count")
    if completed + failed != attempted:
        raise EvaluationArtifactIncompatible("Agent completed and failed denominators differ")
    expected_grounded = _integer(denominators.get("expected_grounded_runs"), "denominators.expected_grounded_runs")
    expected_abstention = _integer(denominators.get("expected_abstention_runs"), "denominators.expected_abstention_runs")
    citations = _integer(denominators.get("citation_count"), "denominators.citation_count")
    expected_references = _integer(denominators.get("expected_relevant_reference_count"), "denominators.expected_relevant_reference_count")
    question_count = _integer(dataset.get("question_count"), "agent.dataset.question_count", 1)
    repeats = _integer(dataset.get("repeats"), "agent.dataset.repeats", 1)
    if attempted != question_count * repeats:
        raise EvaluationArtifactIncompatible("Agent attempted-run denominator differs from dataset")
    definitions = _mapping(data.get("metric_definitions"), "agent.metric_definitions")
    metric_rows = [
        _agent_metric("tool_selection_accuracy", "Tool-selection accuracy", metrics, "attempted runs", attempted, _text(definitions.get("tool_selection_accuracy"), "metric_definitions.tool_selection_accuracy")),
        _agent_metric("document_scope_compliance", "Document-scope compliance", metrics, "attempted runs", attempted, _text(definitions.get("document_scope_compliance"), "metric_definitions.document_scope_compliance")),
        _agent_metric("expected_status_accuracy", "Expected-status accuracy", metrics, "attempted runs", attempted, "Expected terminal status matched the observed status; failed attempts count as incorrect."),
        _agent_metric("grounded_answer_rate", "Grounded-answer rate", metrics, "grounded-labelled runs", expected_grounded, _text(definitions.get("grounded_answer_rate"), "metric_definitions.grounded_answer_rate")),
        _agent_metric("abstention_accuracy", "Abstention accuracy", metrics, "attempted runs", attempted, _text(definitions.get("abstention_accuracy"), "metric_definitions.abstention_accuracy")),
        _agent_metric("unanswerable_abstention_rate", "Unanswerable abstention rate", metrics, "abstention-labelled runs", expected_abstention, "Correct insufficient-evidence outcomes divided by runs labelled for abstention."),
        _agent_metric("citation_validity", "Citation validity", metrics, "returned citations", citations, _text(definitions.get("citation_validity"), "metric_definitions.citation_validity")),
        _agent_metric("citation_coverage", "Citation coverage", metrics, "expected references", expected_references, "Expected supporting references matched by returned citations divided by all expected references."),
    ]
    variability = _mapping(data.get("variability"), "agent.variability")
    usage = _mapping(data.get("usage"), "agent.usage")
    limitations = _mapping(data.get("limitations"), "agent.limitations")
    return {
        "dataset": {
            "version": _text(dataset.get("version"), "agent.dataset.version"),
            "title": _text(dataset.get("title"), "agent.dataset.title"),
            "license": _text(dataset.get("license"), "agent.dataset.license"),
            "document_count": _integer(dataset.get("document_count"), "agent.dataset.document_count", 1),
            "question_count": question_count,
            "repeats": repeats,
            "attempted_run_count": attempted,
            "sha256": _text(dataset.get("sha256"), "agent.dataset.sha256"),
        },
        "run": _run(data.get("run"), "agent.run"),
        "model": {
            "provider": _text(agent.get("provider"), "agent.provider"),
            "model": _text(agent.get("model"), "agent.model"),
            "temperature": _number(agent.get("temperature"), "agent.temperature"),
        },
        "settings": {
            "scope_prompt_version": _text(agent.get("scope_prompt_version"), "agent.scope_prompt_version"),
            "decision_prompt_version": _text(agent.get("document_decision_prompt_version"), "agent.document_decision_prompt_version"),
            "decision_schema_version": _text(agent.get("document_decision_schema_version"), "agent.document_decision_schema_version"),
            "generation_schema_version": _text(agent.get("generation_schema_version"), "agent.generation_schema_version"),
            "graph_step_limit": _integer(agent.get("graph_step_limit"), "agent.graph_step_limit", 1),
            "max_tool_calls": _integer(agent.get("max_tool_calls"), "agent.max_tool_calls", 1),
            "tool_top_k": _integer(agent.get("tool_top_k"), "agent.tool_top_k", 1),
            "execution_timeout_seconds": _number(agent.get("execution_timeout_seconds"), "agent.execution_timeout_seconds"),
            "retrieval_strategy": _text(retrieval.get("strategy"), "agent.retrieval.strategy"),
            "min_similarity": _number(retrieval.get("min_similarity"), "agent.retrieval.min_similarity"),
            "embedding_model": _text(retrieval.get("embedding_model"), "agent.retrieval.embedding_model"),
            "embedding_revision": _text(retrieval.get("embedding_revision"), "agent.retrieval.embedding_revision"),
        },
        "denominators": {
            "attempted_runs": attempted,
            "completed_runs": completed,
            "failed_runs": failed,
            "expected_grounded_runs": expected_grounded,
            "expected_abstention_runs": expected_abstention,
            "citation_count": citations,
            "expected_relevant_reference_count": expected_references,
            "questions": question_count,
        },
        "metrics": metric_rows,
        "latency": _latency(metrics.get("end_to_end_latency_ms"), "agent.metrics.end_to_end_latency_ms"),
        "activity": {
            "tool_calls": _distribution(metrics.get("tool_calls"), "agent.metrics.tool_calls"),
            "graph_steps": _distribution(metrics.get("graph_steps"), "agent.metrics.graph_steps"),
            "stable_question_rate": _ratio(variability.get("stable_question_rate"), "agent.variability.stable_question_rate"),
            "questions_with_variation": _integer(variability.get("questions_with_variation"), "agent.variability.questions_with_variation"),
        },
        "failures": {
            "count": failed,
            "rate": round(failed / attempted, 6),
            "configured_timeout_overrun_count": _integer(metrics.get("configured_timeout_overrun_count"), "metrics.configured_timeout_overrun_count"),
        },
        "usage": {
            "samples_with_usage": _integer(usage.get("samples_with_usage"), "usage.samples_with_usage"),
            "prompt_tokens": _integer(usage.get("prompt_tokens"), "usage.prompt_tokens"),
            "completion_tokens": _integer(usage.get("completion_tokens"), "usage.completion_tokens"),
        },
        "limitations": {
            "citation_check": _text(limitations.get("citation_check"), "limitations.citation_check"),
            "factual_entailment_measured": _boolean(limitations.get("factual_entailment_measured"), "limitations.factual_entailment_measured"),
            "sample_size": _text(limitations.get("sample_size"), "limitations.sample_size"),
        },
        "evidence_observation": None,
        "comparability_note": "Agentic decision metrics use a different dataset, task and denominator from RAG retrieval metrics; cross-panel ranking is invalid.",
    }


def _attach_agent_observation(summary: dict, data: dict) -> None:
    _schema(data, 1, "agent_evidence_observation")
    dataset = _mapping(data.get("dataset"), "observation.dataset")
    if (dataset.get("version") != summary["dataset"]["version"]
            or dataset.get("sha256") != summary["dataset"]["sha256"]):
        raise EvaluationArtifactIncompatible("Agent observation uses another dataset")
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 1 or not isinstance(cases[0], dict):
        raise EvaluationArtifactIncomplete("Agent observation must contain one case")
    case = cases[0]
    trace = case.get("node_trace")
    tools = case.get("tool_activity")
    if not isinstance(trace, list) or not all(isinstance(item, dict) for item in trace):
        raise EvaluationArtifactIncomplete("Agent observation trace is incomplete")
    if not isinstance(tools, list) or not all(isinstance(item, dict) for item in tools):
        raise EvaluationArtifactIncomplete("Agent observation tool activity is incomplete")
    evidence_counts = [
        _integer(item.get("evidence_count"), "observation.evidence_count")
        for item in trace if item.get("evidence_count") is not None
    ]
    result_counts = [
        _integer(item.get("result_count"), "observation.result_count") for item in tools
    ]
    expected = _text(case.get("expected_status"), "observation.expected_status")
    observed = _text(case.get("status"), "observation.status")
    summary["evidence_observation"] = {
        "run": _run(data.get("run"), "observation.run"),
        "sample_size": _integer(dataset.get("attempted_run_count"), "observation.attempted_run_count", 1),
        "expected_status": expected,
        "observed_status": observed,
        "retrieved_result_count": max(result_counts, default=0),
        "evidence_count": max(evidence_counts, default=0),
        "tool_call_count": _integer(case.get("tool_call_count"), "observation.tool_call_count"),
        "finding": "The targeted run retrieved document evidence but the model selected insufficient evidence instead of the expected grounded status.",
        "caution": "One targeted run documents an evidence-assessment limitation; it is not a general performance estimate.",
    }


class EvaluationSummaryService:
    """Expose aggregate, validated research evidence without raw evaluation cases."""

    def __init__(self, reader: EvaluationArtifactReader):
        self.reader = reader

    @staticmethod
    def _issue(issues: list[dict], artifact_id: str, code: str, message: str) -> None:
        issues.append({"artifact_id": artifact_id, "code": code, "message": message})

    def summary(self) -> dict:
        artifacts = {key: {"artifact_id": key, "state": "loaded"} for key in ARTIFACT_IDS}
        issues: list[dict] = []
        raw: dict[str, dict] = {}
        for artifact_id in ARTIFACT_IDS:
            try:
                raw[artifact_id] = self.reader.read(artifact_id)
            except EvaluationArtifactMissing:
                artifacts[artifact_id]["state"] = "missing"
                self._issue(issues, artifact_id, "artifact_missing", "The approved versioned artifact is missing.")
            except (EvaluationArtifactInvalid, EvaluationArtifactAccessDenied):
                artifacts[artifact_id]["state"] = "invalid"
                self._issue(issues, artifact_id, "artifact_invalid", "The approved versioned artifact could not be safely read.")

        rag = None
        if "rag_baseline" in raw:
            try:
                rag = _map_rag_baseline(raw["rag_baseline"])
            except EvaluationArtifactIncompatible:
                artifacts["rag_baseline"]["state"] = "incompatible"
                self._issue(issues, "rag_baseline", "artifact_incompatible", "The RAG baseline schema or denominators are incompatible.")
            except EvaluationArtifactIncomplete:
                artifacts["rag_baseline"]["state"] = "incomplete"
                self._issue(issues, "rag_baseline", "artifact_incomplete", "The RAG baseline is missing required summary data.")
        if rag is not None and "rag_strategy_reference" in raw:
            try:
                _attach_rag_reference(rag, raw["rag_strategy_reference"])
            except EvaluationArtifactIncompatible:
                artifacts["rag_strategy_reference"]["state"] = "incompatible"
                self._issue(issues, "rag_strategy_reference", "artifact_incompatible", "The RAG strategy reference does not match the baseline dataset.")
            except EvaluationArtifactIncomplete:
                artifacts["rag_strategy_reference"]["state"] = "incomplete"
                self._issue(issues, "rag_strategy_reference", "artifact_incomplete", "The RAG strategy reference is missing required summary data.")

        agent = None
        if "agent_baseline" in raw:
            try:
                agent = _map_agent_baseline(raw["agent_baseline"])
            except EvaluationArtifactIncompatible:
                artifacts["agent_baseline"]["state"] = "incompatible"
                self._issue(issues, "agent_baseline", "artifact_incompatible", "The Agentic baseline schema or denominators are incompatible.")
            except EvaluationArtifactIncomplete:
                artifacts["agent_baseline"]["state"] = "incomplete"
                self._issue(issues, "agent_baseline", "artifact_incomplete", "The Agentic baseline is missing required summary data.")
        if agent is not None and "agent_evidence_observation" in raw:
            try:
                _attach_agent_observation(agent, raw["agent_evidence_observation"])
            except EvaluationArtifactIncompatible:
                artifacts["agent_evidence_observation"]["state"] = "incompatible"
                self._issue(issues, "agent_evidence_observation", "artifact_incompatible", "The Agentic observation does not match the baseline dataset.")
            except EvaluationArtifactIncomplete:
                artifacts["agent_evidence_observation"]["state"] = "incomplete"
                self._issue(issues, "agent_evidence_observation", "artifact_incomplete", "The Agentic observation is missing required summary data.")

        findings = []
        if rag is not None:
            semantic = next(item for item in rag["strategies"] if item["strategy"] == "semantic")
            k3 = next((item for item in semantic["metrics"] if item["k"] == 3), None)
            if k3:
                findings.append({
                    "source": "rag_baseline",
                    "title": "RAG retrieval succeeded on the small synthetic reference set",
                    "summary": f"Semantic retrieval reached Recall@3 {k3['recall_at_k']:.3f} and Hit Rate@3 {k3['hit_rate_at_k']:.3f} across {k3['answerable_question_count']} answerable questions.",
                    "caution": "This retrieval result does not measure answer correctness and does not generalize beyond the versioned synthetic dataset.",
                })
        if agent is not None:
            findings.append({
                "source": "agent_baseline",
                "title": "Agent reliability includes explicit failures and separate denominators",
                "summary": f"The baseline completed {agent['denominators']['completed_runs']} of {agent['denominators']['attempted_runs']} attempts; {agent['denominators']['failed_runs']} failures remain in attempted-run accuracy denominators.",
                "caution": "Tool calls, graph steps and citation counts are activity measures, not answer-quality scores.",
            })
            if agent["evidence_observation"]:
                observation = agent["evidence_observation"]
                findings.append({
                    "source": "agent_evidence_observation",
                    "title": "Retrieved evidence did not guarantee a grounded Agentic answer",
                    "summary": f"One targeted run retrieved {observation['retrieved_result_count']} results and retained {observation['evidence_count']} evidence chunks, but observed {observation['observed_status']} where {observation['expected_status']} was expected.",
                    "caution": observation["caution"],
                })

        available = sum(item is not None for item in (rag, agent))
        status = "complete" if available == 2 and not issues else "partial" if available else "unavailable"
        return {
            "schema_version": 1,
            "status": status,
            "artifacts": [artifacts[key] for key in ARTIFACT_IDS],
            "issues": issues,
            "rag": rag,
            "agentic": agent,
            "research_findings": findings,
            "cross_dataset_comparison_allowed": False,
            "generated_from_versioned_artifacts_only": True,
        }
