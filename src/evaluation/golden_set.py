"""
Golden query set management.

The golden set is the authoritative test suite used for offline evaluation
and regression checks before shipping pipeline changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GoldenQuery:
    id: str
    query: str
    cancer_type: str
    difficulty: str = "medium"
    question_type: str = "treatment"
    expected_study_designs: list[str] = field(default_factory=list)
    must_contain_terms: list[str] = field(default_factory=list)
    must_not_hallucinate: bool = True
    ground_truth: str = ""
    expected_sources: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class GoldenSet:
    version: str
    queries: list[GoldenQuery]


def load_golden_set(path: str) -> GoldenSet:
    """Parse and validate the golden set JSON file.

    Supports both v1 (question/topic) and v2 (query/cancer_type) schemas.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    queries: list[GoldenQuery] = []
    for q in data.get("queries", []):
        query_text = q.get("query") or q.get("question", "")
        cancer_type = q.get("cancer_type") or q.get("topic", "")
        queries.append(
            GoldenQuery(
                id=q.get("id", ""),
                query=query_text,
                cancer_type=cancer_type,
                difficulty=q.get("difficulty", "medium"),
                question_type=q.get("question_type", "treatment"),
                expected_study_designs=q.get("expected_study_designs", []),
                must_contain_terms=q.get("must_contain_terms", []),
                must_not_hallucinate=q.get("must_not_hallucinate", True),
                ground_truth=q.get("ground_truth", ""),
                expected_sources=q.get("expected_sources", []),
                notes=q.get("notes", ""),
            )
        )
    return GoldenSet(version=data.get("version", "1.0"), queries=queries)


def add_query(set_path: str, query: GoldenQuery) -> None:
    """Append a query to the golden set file, auto-assigning an ID if needed."""
    path = Path(set_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = {"version": "2.0", "queries": []}

    if not query.id:
        existing_nums = [
            int(q["id"][1:])
            for q in data["queries"]
            if q.get("id", "").startswith("G") and q["id"][1:].isdigit()
        ]
        query.id = f"G{(max(existing_nums, default=0) + 1):03d}"

    data["queries"].append(
        {
            "id": query.id,
            "query": query.query,
            "cancer_type": query.cancer_type,
            "difficulty": query.difficulty,
            "question_type": query.question_type,
            "expected_study_designs": query.expected_study_designs,
            "must_contain_terms": query.must_contain_terms,
            "must_not_hallucinate": query.must_not_hallucinate,
            "ground_truth": query.ground_truth,
            "expected_sources": query.expected_sources,
            "notes": query.notes,
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
