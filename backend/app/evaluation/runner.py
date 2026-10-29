from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import unicodedata
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.application.ingestion import split_pages, validate_vectors
from app.application.rag import RagService
from app.application.retrieval import RetrievalService
from app.domain.documents import Document, PageText
from app.domain.rag import RagResult
from app.evaluation.dataset import EvaluationDataset, EvaluationReference
from app.evaluation.metrics import answer_metrics, percentile, retrieval_metrics
from app.infrastructure.database import Base, create_session_factory
from app.infrastructure.document_repository import SQLAlchemyDocumentRepository
from app.infrastructure.embeddings import create_embedding_provider
from app.infrastructure.llm import create_llm_provider
from app.infrastructure.reranker import create_reranker_provider
from app.infrastructure.retrieval_repository import SQLAlchemyRetrievalRepository


def _canonical(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
            cwd=root, text=True, capture_output=True, check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class EvaluationEnvironment:
    retrieval: RetrievalService
    rag: RagService | None
    document_ids: dict[str, UUID]
    document_keys: dict[UUID, str]
    schema: str


class IsolatedEvaluationDatabase(AbstractContextManager):
    """Transaction-scoped schema; production corpus rows are counted before and after."""

    def __init__(self, settings, dataset: EvaluationDataset, include_llm: bool):
        self.settings = settings
        self.dataset = dataset
        self.include_llm = include_llm
        self.engine = self.connection = self.transaction = self.session = None
        self.public_counts_before: tuple[int, int] | None = None
        self.public_counts_after: tuple[int, int] | None = None

    def _public_counts(self) -> tuple[int, int]:
        with self.engine.connect() as connection:
            return tuple(connection.execute(text(
                "SELECT (SELECT count(*) FROM public.documents), "
                "(SELECT count(*) FROM public.chunks)"
            )).one())

    def __enter__(self) -> EvaluationEnvironment:
        self.engine, _ = create_session_factory(self.settings)
        try:
            self.public_counts_before = self._public_counts()
            self.connection = self.engine.connect()
            self.transaction = self.connection.begin()
            schema = "evaluation_" + uuid4().hex
            self.connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            self.connection.execute(text(f'SET LOCAL search_path TO "{schema}", public'))
            Base.metadata.create_all(self.connection, checkfirst=False)
            self.session = Session(bind=self.connection, join_transaction_mode="create_savepoint")
            embeddings = create_embedding_provider(self.settings)
            document_ids, document_keys = self._seed(embeddings)
            retrieval = RetrievalService(
                SQLAlchemyRetrievalRepository(self.session), embeddings,
                default_top_k=self.settings.retrieval_top_k,
                min_similarity=self.dataset.retrieval_min_similarity,
                default_strategy=self.settings.retrieval_strategy,
                semantic_candidates=self.settings.semantic_candidate_count,
                keyword_candidates=self.settings.keyword_candidate_count,
                rrf_constant=self.settings.rrf_fusion_constant,
                reranker=create_reranker_provider(self.settings),
                reranker_candidates=self.settings.reranker_candidate_count,
            )
            rag = None
            if self.include_llm:
                rag = RagService(retrieval, create_llm_provider(self.settings),
                                 self.settings.rag_max_context_bytes, self.settings.rag_max_prompt_bytes)
            return EvaluationEnvironment(retrieval, rag, document_ids, document_keys, schema)
        except Exception:
            self.__exit__(*sys.exc_info())
            raise

    def _seed(self, embeddings) -> tuple[dict[str, UUID], dict[UUID, str]]:
        repository = SQLAlchemyDocumentRepository(self.session)
        document_ids: dict[str, UUID] = {}
        document_keys: dict[UUID, str] = {}
        actual_references: set[EvaluationReference] = set()
        for specification in self.dataset.documents:
            document_id = uuid5(NAMESPACE_URL, f"{self.dataset.version}:{specification.key}")
            pages = [PageText(index, value) for index, value in enumerate(specification.pages, 1)]
            chunks = split_pages(pages, document_id, self.dataset.chunk_size, self.dataset.chunk_overlap)
            vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
            validate_vectors(vectors, len(chunks), embeddings.dimension)
            serialized = json.dumps(asdict(specification), sort_keys=True, ensure_ascii=False).encode("utf-8")
            document = Document(
                document_id, specification.filename, hashlib.sha256(serialized).hexdigest(), len(serialized),
                len(pages), len(chunks), embeddings.model, embeddings.dimension,
                self.dataset.chunk_size, self.dataset.chunk_overlap, embeddings.profile, embeddings.revision,
            )
            with repository.transaction():
                repository.bind_profile(embeddings.profile)
                repository.add(document, chunks, vectors)
            document_ids[specification.key] = document_id
            document_keys[document_id] = specification.key
            actual_references.update(EvaluationReference(specification.key, chunk.page_number, chunk.chunk_order)
                                     for chunk in chunks)
        expected = {reference for question in self.dataset.questions for reference in question.relevant}
        missing = expected - actual_references
        if missing:
            raise ValueError(f"Dataset references chunks not produced by its pinned chunk configuration: {sorted(missing)}")
        return document_ids, document_keys

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.session is not None:
                self.session.close()
            if self.transaction is not None:
                self.transaction.rollback()
            if self.connection is not None:
                self.connection.close()
            if self.engine is not None and self.public_counts_before is not None:
                self.public_counts_after = self._public_counts()
                if self.public_counts_after != self.public_counts_before and exc_type is None:
                    raise RuntimeError("Evaluation isolation check failed: production corpus counts changed.")
        finally:
            if self.engine is not None:
                self.engine.dispose()
        return False


class EvaluationRunner:
    def __init__(self, dataset: EvaluationDataset, environment: EvaluationEnvironment,
                 settings, run_id: str, repository_root: Path):
        self.dataset = dataset
        self.environment = environment
        self.settings = settings
        self.run_id = run_id
        self.repository_root = repository_root

    def _reference(self, item) -> EvaluationReference:
        key = self.environment.document_keys.get(item.document_id)
        return EvaluationReference(key or "<unknown-document>", item.page_number, item.chunk_order)

    def run(self, ks: list[int], answer_top_k: int, strategies: list[str] | None = None,
            answer_strategy: str = "semantic") -> dict:
        ks = sorted(set(ks))
        if not ks or any(isinstance(k, bool) or not 1 <= k <= 20 for k in ks):
            raise ValueError("Evaluation K values must be unique integers between 1 and 20.")
        if not 1 <= answer_top_k <= 20:
            raise ValueError("answer_top_k must be between 1 and 20.")
        strategies = ["semantic"] if strategies is None else list(dict.fromkeys(strategies))
        allowed = {"semantic", "hybrid", "hybrid_rerank"}
        if not strategies or any(strategy not in allowed for strategy in strategies):
            raise ValueError("Evaluation strategies must be semantic, hybrid or hybrid_rerank.")
        if answer_strategy not in allowed:
            raise ValueError("answer_strategy must be semantic, hybrid or hybrid_rerank.")
        strategy_results = {strategy: self._evaluate_retrieval(ks, strategy) for strategy in strategies}
        primary_name = "semantic" if "semantic" in strategy_results else strategies[0]
        primary = strategy_results[primary_name]
        answer_section = self._evaluate_answers(answer_top_k, answer_strategy)
        reranker = getattr(self.environment.retrieval, "reranker", None)
        metric_definitions = {
            "relevance": "exact match of dataset document key, one-based page and zero-based global chunk order",
            "denominator": "macro average over answerable questions only; unanswerable questions have no relevant-item denominator",
            "recall_at_k": "relevant references in first K / all expected relevant references",
            "precision_at_k": "relevant references in first K / K; missing returned positions count as nonrelevant",
            "mrr_at_k": "reciprocal rank of the first relevant reference within K, or zero",
            "hit_rate_at_k": "fraction with at least one relevant reference within K",
        }
        retrieval_output = {
            "primary_strategy": primary_name,
            "configuration": primary["configuration"],
            "metric_definitions": metric_definitions,
            "metrics": primary["metrics"], "latency_ms": primary["latency_ms"],
            "cases": primary["cases"], "strategy_results": strategy_results,
        }
        return {
            "schema_version": 2,
            "run": {
                "run_id": self.run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_revision": _git_revision(self.repository_root),
                "python": platform.python_version(), "platform": platform.platform(),
                "production_data_mutated": False,
                "isolation": "private transaction-scoped PostgreSQL schema rolled back after the run",
            },
            "dataset": {
                "version": self.dataset.version, "title": self.dataset.title,
                "license": self.dataset.license, "sha256": self.dataset.checksum,
                "document_count": len(self.dataset.documents),
                "question_count": len(self.dataset.questions),
                "answerable_question_count": sum(bool(item.relevant) for item in self.dataset.questions),
                "unanswerable_question_count": sum(not item.relevant for item in self.dataset.questions),
            },
            "embedding": {
                "provider": self.settings.embedding_provider, "model": self.environment.retrieval.embeddings.model,
                "revision": self.environment.retrieval.embeddings.revision,
                "dimension": self.environment.retrieval.embeddings.dimension,
                "profile": self.environment.retrieval.embeddings.profile,
                "normalization": "L2 normalized",
            },
            "reranker": None if reranker is None else {
                "provider": reranker.provider, "model": reranker.model, "revision": reranker.revision,
                "lazy": True, "used": "hybrid_rerank" in strategies or answer_strategy == "hybrid_rerank",
            },
            "retrieval": retrieval_output,
            "answer_evaluation": answer_section,
        }

    def _evaluate_retrieval(self, ks: list[int], strategy: str) -> dict:
        retrieval_cases, retrieval_latencies = [], []
        for question in self.dataset.questions:
            selected = None if question.document_filter is None else [
                self.environment.document_ids[key] for key in question.document_filter
            ]
            result = self.environment.retrieval.retrieve(question.question, max(ks), selected, strategy)
            retrieval_latencies.append(result.retrieval_ms)
            retrieval_cases.append({
                "question_id": question.id,
                "relevant": list(question.relevant),
                "retrieved": [self._reference(chunk) for chunk in result.chunks],
                "details": [{"reference": self._reference(chunk).as_dict(),
                             "distance": round(chunk.distance, 6), "similarity": round(chunk.similarity, 6)}
                            for chunk in result.chunks],
            })
        retrieval_case_output = [{
            "question_id": case["question_id"],
            "relevant": [item.as_dict() for item in case["relevant"]],
            "retrieved": case["details"],
        } for case in retrieval_cases]
        return {
            "configuration": self._retrieval_configuration(strategy, ks),
            "metrics": retrieval_metrics(retrieval_cases, ks),
            "latency_ms": {"sample_count": len(retrieval_latencies),
                           "p50": percentile(retrieval_latencies, 50),
                           "p95": percentile(retrieval_latencies, 95)},
            "cases": retrieval_case_output,
        }

    def _retrieval_configuration(self, strategy: str, ks: list[int]) -> dict:
        base = {
            "distance": "pgvector exact cosine", "k_values": ks,
            "min_similarity": self.dataset.retrieval_min_similarity,
            "chunk_size": self.dataset.chunk_size, "chunk_overlap": self.dataset.chunk_overlap,
        }
        if strategy != "semantic":
            base.update({
                "strategy": strategy,
                "keyword_search": "PostgreSQL english websearch_to_tsquery + ts_rank_cd",
                "semantic_candidate_count": self.environment.retrieval.semantic_candidates,
                "keyword_candidate_count": self.environment.retrieval.keyword_candidates,
                "rrf_fusion_constant": self.environment.retrieval.rrf_constant,
            })
        if strategy == "hybrid_rerank":
            base.update({
                "reranker_candidate_count": self.environment.retrieval.reranker_candidates,
                "reranker_model": self.environment.retrieval.reranker.model,
                "reranker_revision": self.environment.retrieval.reranker.revision,
            })
        return base

    def _evaluate_answers(self, top_k: int, strategy: str) -> dict:
        if self.environment.rag is None:
            return {
                "status": "disabled", "llm_dependent": True,
                "reason": "Run with --with-llm to use the configured local provider.",
                "factual_entailment_measured": False,
            }
        cases, latencies, prompt_tokens, completion_tokens = [], [], [], []
        for question in self.dataset.questions:
            selected = None if question.document_filter is None else [
                self.environment.document_ids[key] for key in question.document_filter
            ]
            try:
                result: RagResult = self.environment.rag.answer(question.question, top_k, selected, strategy)
                latencies.append(result.generation_ms)
                citations = [{
                    "reference": self._reference(citation),
                    "quote_valid": bool(_canonical(citation.quote)) and
                                   _canonical(citation.quote) in _canonical(citation.text),
                } for citation in result.citations]
                cases.append({"question_id": question.id, "relevant": list(question.relevant),
                              "status": result.status, "citations": citations})
                if result.usage and result.usage.prompt_tokens is not None:
                    prompt_tokens.append(result.usage.prompt_tokens)
                if result.usage and result.usage.completion_tokens is not None:
                    completion_tokens.append(result.usage.completion_tokens)
            except Exception as exc:
                cases.append({"question_id": question.id, "relevant": list(question.relevant),
                              "status": "error", "error_type": type(exc).__name__, "citations": []})
        errors = [{"question_id": case["question_id"], "error_type": case["error_type"]}
                  for case in cases if case["status"] == "error"]
        case_output = [{"question_id": case["question_id"], "status": case["status"],
                        "citations": [{"reference": citation["reference"].as_dict(),
                                       "quote_valid": citation["quote_valid"]}
                                      for citation in case["citations"]],
                        **({"error_type": case["error_type"]} if case["status"] == "error" else {})}
                       for case in cases]
        return {
            "status": "completed" if not errors else "partial", "llm_dependent": True,
            "provider": self.environment.rag.llm.provider, "model": self.environment.rag.llm.model,
            "top_k": top_k, "retrieval_strategy": strategy,
            "metric_definitions": {
                "citation_precision": "dataset-relevant citations / all returned citations (micro average)",
                "citation_coverage": "unique expected references cited / all expected references (micro average)",
                "mechanical_quote_validity": "citations whose normalized quote occurs in the supplied chunk / citations",
                "abstention_behavior": "status matches expected answerable/unanswerable label",
                "scope_warning": "source identity and quote presence are mechanical checks; factual entailment and faithfulness are not measured",
            },
            "metrics": answer_metrics(cases),
            "latency_ms": {"sample_count": len(latencies), "p50": percentile(latencies, 50),
                           "p95": percentile(latencies, 95)},
            "token_usage": {
                "prompt_tokens_total": sum(prompt_tokens) if prompt_tokens else None,
                "prompt_usage_sample_count": len(prompt_tokens),
                "completion_tokens_total": sum(completion_tokens) if completion_tokens else None,
                "completion_usage_sample_count": len(completion_tokens),
            },
            "errors": errors, "cases": case_output,
            "factual_entailment_measured": False,
        }


def compare_results(current: dict, baseline: dict) -> dict:
    compatible = (
        current["dataset"]["sha256"] == baseline.get("dataset", {}).get("sha256") and
        current["embedding"]["profile"] == baseline.get("embedding", {}).get("profile") and
        current["retrieval"]["configuration"] == baseline.get("retrieval", {}).get("configuration")
    )
    deltas = {}
    if compatible:
        for k, values in current["retrieval"]["metrics"].items():
            baseline_values = baseline["retrieval"]["metrics"][k]
            deltas[k] = {name: round(values[name] - baseline_values[name], 6)
                         for name in ("recall_at_k", "precision_at_k", "mrr_at_k", "hit_rate_at_k")}
    return {"compatible": compatible, "retrieval_metric_deltas": deltas,
            "regression": compatible and any(value < 0 for group in deltas.values() for value in group.values())}


def compare_strategies(result: dict) -> dict:
    strategies = result["retrieval"]["strategy_results"]
    semantic = strategies.get("semantic")
    if semantic is None:
        return {"reference_strategy": None, "deltas_vs_semantic": {}}
    comparisons = {}
    for strategy, values in strategies.items():
        if strategy == "semantic":
            continue
        metric_deltas = {}
        for k, metrics in values["metrics"].items():
            metric_deltas[k] = {name: round(metrics[name] - semantic["metrics"][k][name], 6)
                                for name in ("recall_at_k", "precision_at_k", "mrr_at_k", "hit_rate_at_k")}
        comparisons[strategy] = {
            "metric_deltas": metric_deltas,
            "p50_latency_delta_ms": round(values["latency_ms"]["p50"] - semantic["latency_ms"]["p50"], 2),
            "p95_latency_delta_ms": round(values["latency_ms"]["p95"] - semantic["latency_ms"]["p95"], 2),
        }
    return {"reference_strategy": "semantic", "deltas_vs_semantic": comparisons}
