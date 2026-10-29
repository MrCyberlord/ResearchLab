from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class EvaluationReference:
    document: str
    page: int
    chunk_order: int

    @classmethod
    def from_dict(cls, value: dict) -> "EvaluationReference":
        return cls(str(value["document"]), int(value["page"]), int(value["chunk_order"]))

    def as_dict(self) -> dict:
        return {"document": self.document, "page": self.page, "chunk_order": self.chunk_order}


@dataclass(frozen=True)
class EvaluationDocument:
    key: str
    filename: str
    pages: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationQuestion:
    id: str
    question: str
    relevant: tuple[EvaluationReference, ...]
    document_filter: tuple[str, ...] | None
    expect_abstain: bool


@dataclass(frozen=True)
class EvaluationDataset:
    version: str
    title: str
    license: str
    chunk_size: int
    chunk_overlap: int
    retrieval_min_similarity: float
    documents: tuple[EvaluationDocument, ...]
    questions: tuple[EvaluationQuestion, ...]
    checksum: str


def default_dataset_path() -> Path:
    return Path(__file__).resolve().parents[2] / "evaluation" / "datasets" / "v1" / "dataset.json"


def load_dataset(path: Path | str) -> EvaluationDataset:
    path = Path(path)
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)
    documents = tuple(EvaluationDocument(str(item["key"]), str(item["filename"]),
                                         tuple(str(page) for page in item["pages"]))
                      for item in payload["documents"])
    questions = tuple(EvaluationQuestion(
        str(item["id"]), str(item["question"]),
        tuple(EvaluationReference.from_dict(ref) for ref in item.get("relevant", [])),
        None if item.get("document_filter") is None else tuple(str(key) for key in item["document_filter"]),
        bool(item["expect_abstain"]),
    ) for item in payload["questions"])
    dataset = EvaluationDataset(
        str(payload["version"]), str(payload["title"]), str(payload["license"]),
        int(payload["chunking"]["size"]), int(payload["chunking"]["overlap"]),
        float(payload["retrieval"]["min_similarity"]), documents, questions,
        hashlib.sha256(raw_bytes).hexdigest(),
    )
    _validate(dataset)
    return dataset


def _validate(dataset: EvaluationDataset) -> None:
    if not dataset.version or not dataset.license or not dataset.documents or not dataset.questions:
        raise ValueError("Evaluation dataset requires version, license, documents and questions.")
    if dataset.chunk_size <= 0 or not 0 <= dataset.chunk_overlap < dataset.chunk_size:
        raise ValueError("Invalid evaluation chunk configuration.")
    document_keys = [document.key for document in dataset.documents]
    question_ids = [question.id for question in dataset.questions]
    if len(set(document_keys)) != len(document_keys) or len(set(question_ids)) != len(question_ids):
        raise ValueError("Evaluation document keys and question IDs must be unique.")
    known = set(document_keys)
    for document in dataset.documents:
        if not document.key or not document.filename or not document.pages or any(not page.strip() for page in document.pages):
            raise ValueError(f"Evaluation document {document.key!r} is incomplete.")
    for question in dataset.questions:
        if not question.question.strip():
            raise ValueError(f"Evaluation question {question.id!r} is empty.")
        if question.expect_abstain != (len(question.relevant) == 0):
            raise ValueError(f"Question {question.id!r} must abstain exactly when it has no relevant references.")
        selected = known if question.document_filter is None else set(question.document_filter)
        if not selected <= known or any(reference.document not in selected for reference in question.relevant):
            raise ValueError(f"Question {question.id!r} contains an invalid document scope or reference.")
        if len(set(question.relevant)) != len(question.relevant):
            raise ValueError(f"Question {question.id!r} contains duplicate relevant references.")
