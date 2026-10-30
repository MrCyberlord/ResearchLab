"""Offline evaluation runner for the real bounded LangGraph document-research agent."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import platform
from pathlib import Path
import subprocess
from time import perf_counter

import psutil

from app.application.agent import AgentService
from app.application.agent_graph import AgentGraph
from app.application.grounding import GROUNDING_PROMPT_VERSION, GroundedAnswerService
from app.domain.rag import RetrievalStrategy
from app.evaluation.agent_dataset import AgentEvaluationDataset
from app.evaluation.metrics import percentile
from app.evaluation.runner import EvaluationEnvironment, _git_revision
from app.infrastructure.agent_llm import (
    AGENT_DOCUMENT_DECISION_PROMPT_VERSION, AGENT_DOCUMENT_DECISION_SCHEMA_VERSION,
    AGENT_GENERATION_SCHEMA_VERSION, AGENT_SCOPE_PROMPT_VERSION, AGENT_TEMPERATURE,
    create_agent_generation_provider,
)
from app.infrastructure.llm import GROUNDED_ANSWER_SCHEMA_VERSION, create_llm_provider


def _canonical(value: str) -> str:
    return " ".join(value.split())


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _distribution(values: list[int]) -> dict:
    return {
        "sample_count": len(values),
        "mean": round(sum(values) / len(values), 3) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _git_worktree_dirty(root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "status", "--porcelain"],
            cwd=root, text=True, capture_output=True, check=True, timeout=5,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def agent_metrics(cases: list[dict]) -> dict:
    """Compute labelled behavior metrics; errors count as incorrect attempts."""
    attempts = len(cases)
    completed = [case for case in cases if case["status"] != "error"]
    expected_grounded = [case for case in cases if case["expected_status"] == "grounded"]
    expected_abstain = [case for case in cases if case["expected_status"] == "insufficient_evidence"]
    citations = [citation for case in cases for citation in case.get("citations", [])]
    expected_references = sum(len(case["relevant"]) for case in expected_grounded)
    covered_references = sum(len(
        set(case["relevant"]) & {citation["reference_key"] for citation in case.get("citations", [])}
    ) for case in expected_grounded)
    tool_correct = sum(case.get("tool_used") == case["expected_tool_use"] for case in cases)
    scope_correct = sum(case.get("actual_scope") == case["expected_scope"] for case in cases)
    status_correct = sum(case["status"] == case["expected_status"] for case in cases)
    abstention_correct = sum(
        case["status"] != "error" and
        ((case["status"] == "insufficient_evidence") ==
         (case["expected_status"] == "insufficient_evidence"))
        for case in cases
    )
    valid_citations = sum(citation["reference_match"] and citation["quote_valid"]
                          for citation in citations)
    return {
        "tool_selection_accuracy": _ratio(tool_correct, attempts),
        "document_scope_compliance": _ratio(scope_correct, attempts),
        "expected_status_accuracy": _ratio(status_correct, attempts),
        "grounded_answer_rate": _ratio(
            sum(case["status"] == "grounded" for case in expected_grounded),
            len(expected_grounded),
        ),
        "abstention_accuracy": _ratio(abstention_correct, attempts),
        "unanswerable_abstention_rate": _ratio(
            sum(case["status"] == "insufficient_evidence" for case in expected_abstain),
            len(expected_abstain),
        ),
        "citation_validity": _ratio(valid_citations, len(citations)),
        "citation_reference_precision": _ratio(
            sum(citation["reference_match"] for citation in citations), len(citations),
        ),
        "citation_coverage": _ratio(covered_references, expected_references),
        "mechanical_quote_validity": _ratio(
            sum(citation["quote_valid"] for citation in citations), len(citations),
        ),
        "tool_calls": _distribution([
            case["tool_call_count"] for case in completed if case.get("tool_call_count") is not None
        ]),
        "graph_steps": _distribution([
            case["graph_steps"] for case in completed if case.get("graph_steps") is not None
        ]),
        "end_to_end_latency_ms": {
            "sample_count": attempts,
            "p50": percentile([case["latency_ms"] for case in cases], 50),
            "p95": percentile([case["latency_ms"] for case in cases], 95),
        },
        "denominators": {
            "attempted_runs": attempts,
            "completed_runs": len(completed),
            "expected_grounded_runs": len(expected_grounded),
            "expected_abstention_runs": len(expected_abstain),
            "citation_count": len(citations),
            "expected_relevant_reference_count": expected_references,
        },
        "error_count": attempts - len(completed),
        "factual_entailment_measured": False,
    }


def decision_variability(cases: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        grouped[case["question_id"]].append(case)
    questions = []
    stable = 0
    for question_id in sorted(grouped):
        signatures = Counter(
            "|".join(map(str, (
                case.get("actual_scope"), case["status"], case.get("tool_used"),
                case.get("tool_call_count"), case.get("error_type"),
            ))) for case in grouped[question_id]
        )
        if len(signatures) == 1:
            stable += 1
        questions.append({
            "question_id": question_id,
            "run_count": len(grouped[question_id]),
            "unique_decision_signatures": len(signatures),
            "signature_counts": dict(sorted(signatures.items())),
        })
    return {
        "question_count": len(grouped),
        "questions_with_variation": len(grouped) - stable,
        "stable_question_rate": _ratio(stable, len(grouped)),
        "signature": "scope|status|tool_used|tool_call_count|error_type",
        "questions": questions,
    }


class EvaluationDocumentSearchTool:
    """Narrow adapter from AgentGraph to the isolated evaluation retrieval service."""

    def __init__(self, environment: EvaluationEnvironment):
        self.environment = environment

    def search(self, query: str, top_k: int, document_ids, strategy: RetrievalStrategy | None):
        return self.environment.retrieval.retrieve(query, top_k, document_ids, strategy)


def build_agent_evaluation_service(environment: EvaluationEnvironment, settings) -> AgentService:
    graph = AgentGraph(
        create_agent_generation_provider(settings),
        EvaluationDocumentSearchTool(environment),
        GroundedAnswerService(
            create_llm_provider(settings), settings.agent_max_context_bytes,
            settings.agent_max_prompt_bytes,
        ),
        max_tool_calls=settings.agent_max_tool_calls,
        max_tool_top_k=settings.agent_tool_top_k,
        max_evidence_chunks=settings.agent_max_evidence_chunks,
        decision_context_bytes=settings.agent_decision_context_bytes,
    )
    return AgentService(
        graph, settings.agent_max_question_chars, settings.agent_graph_step_limit,
        settings.agent_execution_timeout_seconds,
    )


class AgentEvaluationRunner:
    def __init__(self, dataset: AgentEvaluationDataset, environment: EvaluationEnvironment,
                 settings, run_id: str, repository_root: Path,
                 service: AgentService | None = None):
        self.dataset = dataset
        self.environment = environment
        self.settings = settings
        self.run_id = run_id
        self.repository_root = repository_root
        self.service = service or build_agent_evaluation_service(environment, settings)

    def _reference_key(self, citation) -> tuple[str, int, int]:
        return (
            self.environment.document_keys.get(citation.document_id, "<unknown-document>"),
            citation.page_number, citation.chunk_order,
        )

    def run(self, repeats: int = 1, strategy: str = "semantic",
            question_ids: list[str] | None = None) -> dict:
        if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 20:
            raise ValueError("Agent evaluation repeats must be an integer between 1 and 20.")
        if strategy not in {"semantic", "hybrid", "hybrid_rerank"}:
            raise ValueError("Agent evaluation strategy must be semantic, hybrid or hybrid_rerank.")
        known_ids = {question.id for question in self.dataset.questions}
        if question_ids is not None:
            if (not question_ids or len(set(question_ids)) != len(question_ids)
                    or not set(question_ids) <= known_ids):
                raise ValueError("Agent evaluation question IDs must be unique known IDs.")
            selected_ids = set(question_ids)
            questions = tuple(
                question for question in self.dataset.questions if question.id in selected_ids
            )
        else:
            questions = self.dataset.questions
        cases = []
        prompt_tokens = completion_tokens = usage_samples = 0
        for question in questions:
            selected = None if question.document_filter is None else [
                self.environment.document_ids[key] for key in question.document_filter
            ]
            relevant = [(item.document, item.page, item.chunk_order) for item in question.relevant]
            for repeat_index in range(1, repeats + 1):
                started = perf_counter()
                base = {
                    "question_id": question.id, "repeat_index": repeat_index,
                    "tags": list(question.tags), "expected_scope": question.expected_scope,
                    "expected_status": question.expected_status,
                    "expected_tool_use": question.expected_tool_use,
                    "relevant": relevant,
                }
                try:
                    result = self.service.run(
                        question.question, selected, strategy, include_metadata=True,
                    )
                    metadata = result.metadata
                    citations = []
                    for citation in result.citations:
                        reference_key = self._reference_key(citation)
                        quote_valid = bool(_canonical(citation.quote)) and (
                            _canonical(citation.quote) in _canonical(citation.text)
                        )
                        citations.append({
                            "reference": {
                                "document": reference_key[0], "page": reference_key[1],
                                "chunk_order": reference_key[2],
                            },
                            "reference_key": reference_key,
                            "reference_match": reference_key in set(relevant),
                            "quote_valid": quote_valid,
                        })
                    cases.append({
                        **base, "status": result.status,
                        "actual_scope": "general_knowledge" if metadata.route == "direct"
                        else "document_research",
                        "tool_used": metadata.tool_call_count > 0,
                        "tool_call_count": metadata.tool_call_count,
                        "graph_steps": metadata.steps,
                        "latency_ms": round((perf_counter() - started) * 1000, 2),
                        "citations": citations,
                        "tool_activity": [asdict(item) for item in metadata.tool_activity],
                        "node_outcomes": [f"{item.node}:{item.outcome}"
                                          for item in metadata.node_trace],
                    })
                    if metadata.usage is not None:
                        if metadata.usage.prompt_tokens is not None:
                            prompt_tokens += metadata.usage.prompt_tokens
                        if metadata.usage.completion_tokens is not None:
                            completion_tokens += metadata.usage.completion_tokens
                        usage_samples += 1
                except Exception as exc:
                    cases.append({
                        **base, "status": "error", "actual_scope": None,
                        "tool_used": None, "tool_call_count": None, "graph_steps": None,
                        "latency_ms": round((perf_counter() - started) * 1000, 2),
                        "citations": [], "tool_activity": [], "node_outcomes": [],
                        "error_type": type(exc).__name__,
                    })
        provider = self.service.graph.provider
        metrics = agent_metrics(cases)
        metrics["configured_timeout_overrun_count"] = sum(
            case["latency_ms"] > self.service.execution_timeout_seconds * 1000
            for case in cases
        )
        return {
            "schema_version": 1,
            "run": {
                "run_id": self.run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_revision": _git_revision(self.repository_root),
                "git_worktree_dirty": _git_worktree_dirty(self.repository_root),
                "python": platform.python_version(), "platform": platform.platform(),
                "hardware": {
                    "processor": platform.processor() or None,
                    "physical_cpu_cores": psutil.cpu_count(logical=False),
                    "logical_cpu_count": psutil.cpu_count(logical=True),
                    "visible_memory_gib": round(psutil.virtual_memory().total / (1024 ** 3), 2),
                    "available_memory_gib_at_report": round(
                        psutil.virtual_memory().available / (1024 ** 3), 2
                    ),
                },
                "production_data_mutated": False,
                "isolation": "private transaction-scoped PostgreSQL schema rolled back after the run",
            },
            "dataset": {
                "version": self.dataset.version, "title": self.dataset.title,
                "license": self.dataset.license, "sha256": self.dataset.checksum,
                "document_count": len(self.dataset.documents),
                "question_count": len(questions),
                "total_question_count": len(self.dataset.questions),
                "selected_question_ids": None if question_ids is None else [
                    question.id for question in questions
                ],
                "repeats": repeats,
                "attempted_run_count": len(cases),
                "category_counts": dict(sorted(Counter(
                    tag for question in questions for tag in question.tags
                ).items())),
            },
            "agent": {
                "provider": provider.provider, "model": provider.model,
                "temperature": AGENT_TEMPERATURE,
                "scope_prompt_version": AGENT_SCOPE_PROMPT_VERSION,
                "document_decision_prompt_version": AGENT_DOCUMENT_DECISION_PROMPT_VERSION,
                "document_decision_schema_version": AGENT_DOCUMENT_DECISION_SCHEMA_VERSION,
                "generation_schema_version": AGENT_GENERATION_SCHEMA_VERSION,
                "grounding_prompt_version": GROUNDING_PROMPT_VERSION,
                "grounded_answer_schema_version": GROUNDED_ANSWER_SCHEMA_VERSION,
                "graph_step_limit": self.service.graph_step_limit,
                "execution_timeout_seconds": self.service.execution_timeout_seconds,
                "max_tool_calls": self.service.graph.max_tool_calls,
                "tool_top_k": self.service.graph.max_tool_top_k,
            },
            "retrieval": {
                "strategy": strategy,
                "embedding_model": self.environment.retrieval.embeddings.model,
                "embedding_revision": self.environment.retrieval.embeddings.revision,
                "embedding_profile": self.environment.retrieval.embeddings.profile,
                "min_similarity": self.dataset.retrieval_min_similarity,
            },
            "metric_definitions": {
                "tool_selection_accuracy": "expected tool-use boolean matched / all attempted runs; errors are incorrect",
                "document_scope_compliance": "expected general/document scope matched / all attempted runs; errors are incorrect",
                "grounded_answer_rate": "grounded responses / runs labelled grounded",
                "abstention_accuracy": "expected versus observed insufficient-evidence behavior / all attempted runs",
                "citation_validity": "citations with an expected document/page/chunk identity and verbatim quote containment / all citations",
                "latency": "nearest-rank p50/p95 wall-clock AgentService latency, including failed attempts",
                "configured_timeout_overrun_count": "attempts whose wall-clock latency exceeded the configured total execution timeout",
                "variability": "unique scope/status/tool-use/tool-count/error signatures across repeated runs",
            },
            "metrics": metrics,
            "variability": decision_variability(cases),
            "usage": {
                "samples_with_usage": usage_samples,
                "prompt_tokens": prompt_tokens if usage_samples else None,
                "completion_tokens": completion_tokens if usage_samples else None,
            },
            "cases": cases,
            "limitations": {
                "factual_entailment_measured": False,
                "citation_check": "mechanical source identity and verbatim quote containment only",
                "sample_size": "small synthetic dataset; not representative of production domains",
            },
        }


def compare_agent_results(current: dict, baseline: dict) -> dict:
    current_values = (
        current.get("dataset", {}).get("sha256"), current.get("dataset", {}).get("repeats"),
        tuple(current.get("dataset", {}).get("selected_question_ids") or ("<all>",)),
        current.get("agent", {}).get("provider"), current.get("agent", {}).get("model"),
        current.get("agent", {}).get("scope_prompt_version"),
        current.get("agent", {}).get("document_decision_prompt_version"),
        current.get("agent", {}).get("document_decision_schema_version"),
        current.get("agent", {}).get("generation_schema_version"),
        current.get("agent", {}).get("grounding_prompt_version"),
        current.get("agent", {}).get("grounded_answer_schema_version"),
        current.get("agent", {}).get("temperature"), current.get("retrieval", {}).get("strategy"),
        current.get("retrieval", {}).get("embedding_profile"),
        current.get("retrieval", {}).get("min_similarity"),
        current.get("agent", {}).get("graph_step_limit"),
        current.get("agent", {}).get("max_tool_calls"),
        current.get("agent", {}).get("tool_top_k"),
    )
    baseline_values = (
        baseline.get("dataset", {}).get("sha256"), baseline.get("dataset", {}).get("repeats"),
        tuple(baseline.get("dataset", {}).get("selected_question_ids") or ("<all>",)),
        baseline.get("agent", {}).get("provider"), baseline.get("agent", {}).get("model"),
        baseline.get("agent", {}).get("scope_prompt_version"),
        baseline.get("agent", {}).get("document_decision_prompt_version"),
        baseline.get("agent", {}).get("document_decision_schema_version"),
        baseline.get("agent", {}).get("generation_schema_version"),
        baseline.get("agent", {}).get("grounding_prompt_version"),
        baseline.get("agent", {}).get("grounded_answer_schema_version"),
        baseline.get("agent", {}).get("temperature"), baseline.get("retrieval", {}).get("strategy"),
        baseline.get("retrieval", {}).get("embedding_profile"),
        baseline.get("retrieval", {}).get("min_similarity"),
        baseline.get("agent", {}).get("graph_step_limit"),
        baseline.get("agent", {}).get("max_tool_calls"),
        baseline.get("agent", {}).get("tool_top_k"),
    )
    compatible = all(value is not None for value in (*current_values, *baseline_values)) and (
        current_values == baseline_values
    )
    if not compatible:
        return {"compatible": False, "metric_deltas": {}, "regression": False}
    names = [
        "tool_selection_accuracy", "document_scope_compliance", "expected_status_accuracy",
        "grounded_answer_rate", "abstention_accuracy", "unanswerable_abstention_rate",
        "citation_validity", "citation_reference_precision", "citation_coverage",
        "mechanical_quote_validity",
    ]
    deltas = {}
    for name in names:
        current_value = current["metrics"].get(name)
        baseline_value = baseline["metrics"].get(name)
        if current_value is not None and baseline_value is not None:
            deltas[name] = round(current_value - baseline_value, 6)
    failure_deltas = {
        "error_count": current["metrics"]["error_count"] - baseline["metrics"]["error_count"],
        "configured_timeout_overrun_count": (
            current["metrics"]["configured_timeout_overrun_count"] -
            baseline["metrics"]["configured_timeout_overrun_count"]
        ),
    }
    return {
        "compatible": True, "metric_deltas": deltas, "failure_count_deltas": failure_deltas,
        "regression": (any(value < 0 for value in deltas.values()) or
                       any(value > 0 for value in failure_deltas.values())),
    }
