"""CLI for rollback-isolated, real-model LangGraph agent evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from app.evaluation.agent_dataset import default_agent_dataset_path, load_agent_dataset
from app.evaluation.agent_runner import AgentEvaluationRunner, compare_agent_results
from app.evaluation.runner import IsolatedEvaluationDatabase
from app.infrastructure.config import Settings


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_agent_baseline_path() -> Path:
    return (Path(__file__).resolve().parents[2] / "evaluation" / "agent" /
            "baselines" / "gemma3-4b-v1.json")


def _display(value) -> str:
    return "n/a" if value is None else str(value)


def report(result: dict) -> str:
    metrics = result["metrics"]
    denominators = metrics["denominators"]
    hardware = result["run"].get("hardware", {})
    hardware_summary = (
        f"Hardware: `{hardware.get('processor') or 'unknown CPU'}`; "
        f"{_display(hardware.get('physical_cpu_cores'))} physical / "
        f"{_display(hardware.get('logical_cpu_count'))} logical CPUs; "
        f"{_display(hardware.get('visible_memory_gib'))} GiB visible RAM; "
        f"{_display(hardware.get('available_memory_gib_at_report'))} GiB available at report."
    )
    lines = [
        f"# ResearchLab agent evaluation - {result['run']['run_id']}", "",
        f"Dataset: `{result['dataset']['version']}`; {result['dataset']['question_count']} questions x "
        f"{result['dataset']['repeats']} repeats = {result['dataset']['attempted_run_count']} attempted runs.",
        f"Model: `{result['agent']['provider']}/{result['agent']['model']}` at temperature "
        f"{result['agent']['temperature']}; retrieval strategy `{result['retrieval']['strategy']}`.",
        hardware_summary,
        f"Scope prompt `{result['agent']['scope_prompt_version']}`; document schema "
        f"`{result['agent']['document_decision_schema_version']}`.", "",
        "| Metric | Value | Denominator |",
        "| --- | ---: | ---: |",
        f"| Tool-selection accuracy | {_display(metrics['tool_selection_accuracy'])} | "
        f"{denominators['attempted_runs']} attempted runs |",
        f"| Document-scope compliance | {_display(metrics['document_scope_compliance'])} | "
        f"{denominators['attempted_runs']} attempted runs |",
        f"| Expected-status accuracy | {_display(metrics['expected_status_accuracy'])} | "
        f"{denominators['attempted_runs']} attempted runs |",
        f"| Grounded-answer rate | {_display(metrics['grounded_answer_rate'])} | "
        f"{denominators['expected_grounded_runs']} grounded-labelled runs |",
        f"| Abstention accuracy | {_display(metrics['abstention_accuracy'])} | "
        f"{denominators['attempted_runs']} attempted runs |",
        f"| Unanswerable abstention rate | {_display(metrics['unanswerable_abstention_rate'])} | "
        f"{denominators['expected_abstention_runs']} abstention-labelled runs |",
        f"| Citation validity | {_display(metrics['citation_validity'])} | "
        f"{denominators['citation_count']} returned citations |",
        f"| Citation coverage | {_display(metrics['citation_coverage'])} | "
        f"{denominators['expected_relevant_reference_count']} expected references |",
        "",
        f"End-to-end latency: p50 `{metrics['end_to_end_latency_ms']['p50']}` ms; "
        f"p95 `{metrics['end_to_end_latency_ms']['p95']}` ms.",
        f"Configured-timeout overruns: `{metrics['configured_timeout_overrun_count']}` attempts.",
        f"Completed runs: `{denominators['completed_runs']}`; errors: `{metrics['error_count']}`.",
        f"Tool calls: mean `{metrics['tool_calls']['mean']}`, range "
        f"`{metrics['tool_calls']['min']}..{metrics['tool_calls']['max']}`. "
        f"Graph steps: mean `{metrics['graph_steps']['mean']}`, range "
        f"`{metrics['graph_steps']['min']}..{metrics['graph_steps']['max']}`.",
        f"Decision variability: {result['variability']['questions_with_variation']} of "
        f"{result['variability']['question_count']} questions varied across repeats; stable-question rate "
        f"`{result['variability']['stable_question_rate']}`.", "",
        "## Cases", "",
        "| Question | Repeat | Expected | Actual | Tool calls | Citations | Latency ms | Error |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for case in result["cases"]:
        expected = f"{case['expected_scope']}/{case['expected_status']}"
        actual = f"{case.get('actual_scope')}/{case['status']}"
        lines.append(
            f"| {case['question_id']} | {case['repeat_index']} | {expected} | {actual} | "
            f"{_display(case.get('tool_call_count'))} | {len(case.get('citations', []))} | "
            f"{case['latency_ms']} | {case.get('error_type', '')} |"
        )
    lines += [
        "", "## Interpretation limits", "",
        "Citation validity checks only expected document/page/chunk identity and verbatim quote containment. "
        "It does not measure factual entailment, semantic faithfulness or completeness.",
        "This small synthetic dataset is suitable for reproducible regression signals, not claims about "
        "production-domain quality. Timing and local-model decisions depend on hardware and runtime state.",
    ]
    if "baseline_comparison" in result:
        comparison = result["baseline_comparison"]
        lines += [
            "", f"Baseline comparison: {'compatible' if comparison['compatible'] else 'incompatible'}; "
            f"regression={comparison['regression']}.",
        ]
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Evaluate ResearchLab's real LangGraph document-research agent.",
    )
    command.add_argument("--dataset", type=Path, default=default_agent_dataset_path())
    command.add_argument("--repeats", type=int, default=1)
    command.add_argument(
        "--question-id", action="append", default=None,
        help="Evaluate only this question ID (repeat for a small targeted reliability run).",
    )
    command.add_argument("--strategy", choices=["semantic", "hybrid", "hybrid_rerank"],
                         default="semantic")
    command.add_argument("--run-id", default=None)
    command.add_argument("--output", type=Path,
                         default=Path(".cache/evaluations/agent-latest.json"))
    command.add_argument("--report", type=Path,
                         default=Path(".cache/evaluations/agent-latest.md"))
    command.add_argument("--baseline", type=Path, default=default_agent_baseline_path())
    command.add_argument("--fail-on-regression", action="store_true")
    return command


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    settings = Settings()
    dataset = load_agent_dataset(arguments.dataset)
    run_id = arguments.run_id or datetime.now(timezone.utc).strftime("agent-%Y%m%dT%H%M%SZ")
    manager = IsolatedEvaluationDatabase(settings, dataset, include_llm=False)
    with manager as environment:
        result = AgentEvaluationRunner(
            dataset, environment, settings, run_id, repository_root(),
        ).run(arguments.repeats, arguments.strategy, arguments.question_id)
    result["run"]["production_counts_before"] = list(manager.public_counts_before)
    result["run"]["production_counts_after"] = list(manager.public_counts_after)
    result["run"]["production_data_mutated"] = (
        manager.public_counts_before != manager.public_counts_after
    )
    if arguments.baseline.exists():
        result["baseline_comparison"] = compare_agent_results(
            result, json.loads(arguments.baseline.read_text("utf-8")),
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    arguments.report.write_text(report(result), encoding="utf-8")
    print(report(result), end="")
    print(f"JSON: {arguments.output}\nReport: {arguments.report}")
    comparison = result.get("baseline_comparison")
    if arguments.fail_on_regression and comparison and (
        not comparison["compatible"] or comparison["regression"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
