"""
Evaluation pipeline runner.

Orchestrates a full evaluation pass over the golden set, parallelising
judge calls to minimise wall-clock time, and persisting results to disk
in a format that can be loaded by the Streamlit dashboard.

Run modes:
    full        — all golden queries, all judges (default)
    quick       — random sample of 10 queries, faithfulness + relevance only
    regression  — compare current run against a saved baseline;
                  fail if overall drops by > 2 pp on any metric

Output files:
    data/evaluation/
        {timestamp}_detailed.json     — per-query scores and answers
        {timestamp}_metrics.json      — aggregate metrics matching the
                                        format expected by the Streamlit dashboard
        latest_metrics.json           — symlink/copy of the most recent run

Public API (to be implemented):
    def run_evaluation(
        retriever: Retriever,
        golden_set_path: str = "tests/fixtures/golden_queries.json",
        output_dir: str = "data/evaluation",
        mode: Literal["full", "quick", "regression"] = "full",
        baseline_path: str | None = None,
        max_workers: int = 4,
    ) -> EvaluationReport:

    EvaluationReport(dataclass)
        run_id: str
        timestamp: str
        golden_set_version: str
        mode: str
        total_queries: int
        aggregate: JudgeScores               # mean across all queries
        per_topic: dict[str, JudgeScores]
        per_difficulty: dict[str, JudgeScores]
        per_question_type: dict[str, JudgeScores]
        latency_stats: dict[str, float]      # p50, p95, p99, mean
        regression_delta: JudgeScores | None # vs baseline, None if no baseline
        passed_regression: bool | None
"""
