from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation.dataset import default_dataset_path, load_dataset
from app.evaluation.runner import (EvaluationRunner, IsolatedEvaluationDatabase,
                                   compare_results, compare_strategies)
from app.infrastructure.config import Settings


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_baseline_path() -> Path:
    return Path(__file__).resolve().parents[2] / "evaluation" / "baselines" / "minilm-v1.json"


def report(result: dict) -> str:
    lines = [
        f"# ResearchLab RAG evaluation - {result['run']['run_id']}", "",
        f"Dataset: `{result['dataset']['version']}` ({result['dataset']['question_count']} questions; "
        f"{result['dataset']['unanswerable_question_count']} unanswerable)",
        f"Embedding: `{result['embedding']['model']}@{result['embedding']['revision']}`; "
        f"profile `{result['embedding']['profile']}`", "",
        "| Strategy | K | Recall@K | Precision@K | MRR@K | Hit Rate@K | p50 ms | p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, strategy_result in result["retrieval"]["strategy_results"].items():
        latency = strategy_result["latency_ms"]
        for k, values in strategy_result["metrics"].items():
            lines.append(f"| {strategy} | {k} | {values['recall_at_k']:.3f} | {values['precision_at_k']:.3f} | "
                         f"{values['mrr_at_k']:.3f} | {values['hit_rate_at_k']:.3f} | "
                         f"{latency['p50']} | {latency['p95']} |")
    answers = result["answer_evaluation"]
    if answers["status"] == "disabled":
        lines += ["", "Answer evaluation: disabled (optional; requires `--with-llm`)."]
    else:
        metrics = answers["metrics"]
        lines += ["", f"Answer evaluation: {answers['status']} using `{answers['provider']}/{answers['model']}`.",
                  f"Citation precision {metrics['citation_precision']}; citation coverage {metrics['citation_coverage']}; "
                  f"abstention behavior accuracy {metrics['abstention_behavior_accuracy']}."]
    lines += ["", "Citation identity and quote presence are mechanically checked. This evaluation does not measure "
                  "factual entailment or claim faithfulness."]
    if "baseline_comparison" in result:
        comparison = result["baseline_comparison"]
        lines += ["", f"Baseline comparison: {'compatible' if comparison['compatible'] else 'incompatible'}; "
                        f"regression={comparison['regression']}."]
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Evaluate ResearchLab retrieval and optional grounded answers.")
    command.add_argument("--dataset", type=Path, default=default_dataset_path())
    command.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    command.add_argument("--answer-top-k", type=int, default=5)
    command.add_argument("--strategies", nargs="+", choices=["semantic", "hybrid", "hybrid_rerank"],
                         default=["semantic"])
    command.add_argument("--answer-strategy", choices=["semantic", "hybrid", "hybrid_rerank"],
                         default="semantic")
    command.add_argument("--with-llm", action="store_true", help="Run optional local Ollama answer evaluation.")
    command.add_argument("--run-id", default=None)
    command.add_argument("--output", type=Path, default=Path(".cache/evaluations/latest.json"))
    command.add_argument("--report", type=Path, default=Path(".cache/evaluations/latest.md"))
    command.add_argument("--baseline", type=Path, default=default_baseline_path())
    command.add_argument("--fail-on-regression", action="store_true")
    return command


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    settings = Settings()
    dataset = load_dataset(arguments.dataset)
    run_id = arguments.run_id or datetime.now(timezone.utc).strftime("local-%Y%m%dT%H%M%SZ")
    environment_manager = IsolatedEvaluationDatabase(settings, dataset, arguments.with_llm)
    with environment_manager as environment:
        result = EvaluationRunner(dataset, environment, settings, run_id, repository_root()).run(
            arguments.k, arguments.answer_top_k, arguments.strategies, arguments.answer_strategy
        )
    result["run"]["production_counts_before"] = list(environment_manager.public_counts_before)
    result["run"]["production_counts_after"] = list(environment_manager.public_counts_after)
    result["strategy_comparison"] = compare_strategies(result)
    if arguments.baseline.exists():
        result["baseline_comparison"] = compare_results(result, json.loads(arguments.baseline.read_text("utf-8")))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    arguments.report.write_text(report(result), encoding="utf-8")
    print(report(result), end="")
    print(f"JSON: {arguments.output}\nReport: {arguments.report}")
    comparison = result.get("baseline_comparison")
    if arguments.fail_on_regression and comparison and (not comparison["compatible"] or comparison["regression"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
