"""
Golden query set management.

The golden set is the authoritative test suite used for offline evaluation
and regression checks before shipping pipeline changes.

Golden query schema (`tests/fixtures/golden_queries.json`):
    {
      "version": "1.0",
      "queries": [
        {
          "id": "G001",
          "question": "...",
          "topic": "prostate | bladder | kidney | testicular",
          "difficulty": "easy | medium | hard",
          "question_type": "treatment | diagnosis | prognosis | mechanism | safety",
          "ground_truth": "...",
          "expected_sources": ["PMID:...", ...],   # optional
          "notes": "..."                           # optional curation notes
        }
      ]
    }

Versioning:
    The golden set is versioned alongside the codebase.  When queries are
    added or ground truths updated, the `version` field must be bumped.
    Evaluation results reference the golden set version for traceability.

Public API (to be implemented):
    def load_golden_set(path: str) -> GoldenSet:
        Parse and validate the JSON file.

    def add_query(set_path: str, query: GoldenQuery) -> None:
        Append a new query, auto-assign ID, and save.

    GoldenSet(dataclass)
        version: str
        queries: list[GoldenQuery]

    GoldenQuery(dataclass)
        id: str
        question: str
        topic: str
        difficulty: str
        question_type: str
        ground_truth: str
        expected_sources: list[str]
        notes: str
"""
