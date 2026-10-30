"""Versioned, explicitly labelled dataset for document-research agent evaluation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.evaluation.dataset import EvaluationDocument, EvaluationReference

AgentExpectedScope = Literal["general_knowledge", "document_research"]
AgentExpectedStatus = Literal["direct", "grounded", "insufficient_evidence"]


@dataclass(frozen=True)
class AgentEvaluationQuestion:
    id: str
    question: str
    expected_scope: AgentExpectedScope
    expected_status: AgentExpectedStatus
    expected_tool_use: bool
    relevant: tuple[EvaluationReference, ...]
    document_filter: tuple[str, ...] | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class AgentEvaluationDataset:
    version: str
    title: str
    license: str
    chunk_size: int
    chunk_overlap: int
    retrieval_min_similarity: float
    documents: tuple[EvaluationDocument, ...]
    questions: tuple[AgentEvaluationQuestion, ...]
    checksum: str


def default_agent_dataset_path() -> Path:
    return (Path(__file__).resolve().parents[2] / "evaluation" / "agent" /
            "datasets" / "v1" / "dataset.json")


def load_agent_dataset(path: Path | str) -> AgentEvaluationDataset:
    path = Path(path)
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)
    documents = tuple(EvaluationDocument(
        str(item["key"]), str(item["filename"]),
        tuple(str(page) for page in item["pages"]),
    ) for item in payload["documents"])
    questions = []
    for item in payload["questions"]:
        if type(item.get("expected_tool_use")) is not bool:
            raise ValueError("Agent evaluation expected_tool_use must be boolean.")
        questions.append(AgentEvaluationQuestion(
            str(item["id"]), str(item["question"]), str(item["expected_scope"]),
            str(item["expected_status"]), item["expected_tool_use"],
            tuple(EvaluationReference.from_dict(ref) for ref in item.get("relevant", [])),
            None if item.get("document_filter") is None else
            tuple(str(key) for key in item["document_filter"]),
            tuple(str(tag) for tag in item.get("tags", [])),
        ))
    dataset = AgentEvaluationDataset(
        str(payload["version"]), str(payload["title"]), str(payload["license"]),
        int(payload["chunking"]["size"]), int(payload["chunking"]["overlap"]),
        float(payload["retrieval"]["min_similarity"]), documents, tuple(questions),
        hashlib.sha256(raw_bytes).hexdigest(),
    )
    _validate(dataset)
    return dataset


def _validate(dataset: AgentEvaluationDataset) -> None:
    if not dataset.version or not dataset.license or not dataset.documents or not dataset.questions:
        raise ValueError("Agent evaluation requires version, license, documents and questions.")
    if dataset.chunk_size <= 0 or not 0 <= dataset.chunk_overlap < dataset.chunk_size:
        raise ValueError("Invalid agent evaluation chunk configuration.")
    document_keys = [document.key for document in dataset.documents]
    question_ids = [question.id for question in dataset.questions]
    if len(set(document_keys)) != len(document_keys) or len(set(question_ids)) != len(question_ids):
        raise ValueError("Agent evaluation document keys and question IDs must be unique.")
    known = set(document_keys)
    for document in dataset.documents:
        if (not document.key or not document.filename or not document.pages
                or any(not page.strip() for page in document.pages)):
            raise ValueError(f"Agent evaluation document {document.key!r} is incomplete.")
    for question in dataset.questions:
        if not question.question.strip() or not question.tags:
            raise ValueError(f"Agent evaluation question {question.id!r} is incomplete.")
        if question.expected_scope not in {"general_knowledge", "document_research"}:
            raise ValueError(f"Question {question.id!r} has an invalid expected scope.")
        if question.expected_status not in {"direct", "grounded", "insufficient_evidence"}:
            raise ValueError(f"Question {question.id!r} has an invalid expected status.")
        if question.expected_scope == "general_knowledge":
            if question.expected_status != "direct" or question.expected_tool_use or question.relevant:
                raise ValueError(f"General question {question.id!r} has inconsistent labels.")
        elif not question.expected_tool_use or question.expected_status == "direct":
            raise ValueError(f"Document question {question.id!r} has inconsistent tool labels.")
        if (question.expected_status == "grounded") != bool(question.relevant):
            raise ValueError(f"Question {question.id!r} has inconsistent relevance labels.")
        selected = known if question.document_filter is None else set(question.document_filter)
        if not selected <= known or any(reference.document not in selected for reference in question.relevant):
            raise ValueError(f"Question {question.id!r} has an invalid document scope or reference.")
        if len(set(question.relevant)) != len(question.relevant):
            raise ValueError(f"Question {question.id!r} contains duplicate references.")
    required_tags = {
        "general_knowledge", "document_grounded", "unanswerable_document",
        "paraphrase", "ambiguous", "repeated_search",
    }
    actual_tags = {tag for question in dataset.questions for tag in question.tags}
    if not required_tags <= actual_tags:
        raise ValueError(f"Agent evaluation is missing required scenarios: {sorted(required_tags - actual_tags)}")
