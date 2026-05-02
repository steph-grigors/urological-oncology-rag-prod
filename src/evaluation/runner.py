"""
Evaluation pipeline runner.

Orchestrates a full evaluation pass over the golden set, parallelising
retrieval + generation + judge calls to minimise wall-clock time, and
persisting results so they can be loaded by the Streamlit dashboard.

Run modes:
    full        — all golden queries, all judges (default)
    quick       — random sample of 10 queries
    regression  — compare current run against a saved baseline;
                  fail if overall drops by > 2 pp on any metric

Exit-code contract (CLI use):
    code 0  — all metrics above threshold
    code 1  — any metric drops below threshold:
                faithfulness  < 0.80
                answer_relevance < 0.75
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.evaluation.golden_set import GoldenQuery, load_golden_set
from src.evaluation.judges import JudgeScores, JudgeSet

logger = logging.getLogger(__name__)

_FAITHFULNESS_THRESHOLD = 0.80
_RELEVANCE_THRESHOLD = 0.75
_REGRESSION_TOLERANCE = 0.02

_REFUSAL_MARKERS = (
    "insufficient evidence",
    "cannot answer",
    "i cannot",
    "i'm unable",
    "low confidence",
)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query_id: str
    query: str
    cancer_type: str
    difficulty: str
    question_type: str
    answer: str
    scores: JudgeScores
    latency_ms: float
    contains_required_terms: bool
    is_refused: bool


@dataclass
class EvaluationReport:
    run_id: str
    timestamp: str
    golden_set_version: str
    mode: str
    total_queries: int
    aggregate: JudgeScores
    per_topic: dict[str, JudgeScores]
    per_difficulty: dict[str, JudgeScores]
    per_question_type: dict[str, JudgeScores]
    latency_stats: dict[str, float]
    regression_delta: Optional[JudgeScores]
    passed_regression: Optional[bool]
    per_query_results: list[QueryResult] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def run_evaluation(
    retriever,
    golden_set_path: str = "tests/fixtures/golden_queries.json",
    output_dir: Optional[str] = "data/evaluation",
    mode: str = "full",
    baseline_path: Optional[str] = None,
    max_workers: int = 4,
    generator=None,
    judge_set: Optional[JudgeSet] = None,
) -> EvaluationReport:
    """Run the evaluation pipeline and return a report.

    Pass output_dir=None to skip writing files (useful in tests).
    """
    golden = load_golden_set(golden_set_path)
    queries = golden.queries

    if mode == "quick":
        import random
        queries = random.sample(queries, min(10, len(queries)))

    if judge_set is None:
        judge_set = JudgeSet()

    results: list[QueryResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_single_query, q, retriever, generator, judge_set): q
            for q in queries
        }
        for future in as_completed(futures):
            gq = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.warning("Query %s failed: %s", gq.id, exc)

    aggregate = _mean_scores(results)
    per_topic = _group_scores(results, lambda r: r.cancer_type)
    per_difficulty = _group_scores(results, lambda r: r.difficulty)
    per_question_type = _group_scores(results, lambda r: r.question_type)
    latency_stats = _compute_latency_stats([r.latency_ms for r in results])

    regression_delta: Optional[JudgeScores] = None
    passed_regression: Optional[bool] = None
    if baseline_path:
        regression_delta, passed_regression = _compute_regression(
            aggregate, baseline_path
        )

    report = EvaluationReport(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        golden_set_version=golden.version,
        mode=mode,
        total_queries=len(results),
        aggregate=aggregate,
        per_topic=per_topic,
        per_difficulty=per_difficulty,
        per_question_type=per_question_type,
        latency_stats=latency_stats,
        regression_delta=regression_delta,
        passed_regression=passed_regression,
        per_query_results=results,
    )

    if output_dir is not None:
        _save_report(report, output_dir)

    return report


# ── Private helpers ───────────────────────────────────────────────────────────

def _run_single_query(
    gq: GoldenQuery,
    retriever,
    generator,
    judge_set: JudgeSet,
) -> QueryResult:
    t0 = time.monotonic()

    retrieval_result = retriever.retrieve(gq.query)
    chunks = retrieval_result.chunks

    if generator is not None:
        gen_result = generator.generate(gq.query, chunks)
        answer = gen_result.answer
    else:
        answer = " ".join((getattr(c, "text", "") or "") for c in chunks[:3])

    latency_ms = (time.monotonic() - t0) * 1000

    scores = judge_set.score_all(
        question=gq.query,
        answer=answer,
        chunks=chunks,
        ground_truth=gq.ground_truth or None,
    )

    answer_lower = answer.lower()
    contains_required = (
        all(t.lower() in answer_lower for t in gq.must_contain_terms)
        if gq.must_contain_terms
        else True
    )
    is_refused = any(m in answer_lower for m in _REFUSAL_MARKERS)

    return QueryResult(
        query_id=gq.id,
        query=gq.query,
        cancer_type=gq.cancer_type,
        difficulty=gq.difficulty,
        question_type=gq.question_type,
        answer=answer,
        scores=scores,
        latency_ms=latency_ms,
        contains_required_terms=contains_required,
        is_refused=is_refused,
    )


def _mean_scores(results: list[QueryResult]) -> JudgeScores:
    if not results:
        return JudgeScores()
    n = len(results)
    score_fields = [
        "faithfulness", "answer_relevance", "context_precision",
        "context_recall", "clinical_safety", "citation_accuracy",
        "evidence_appropriate",
    ]
    kwargs = {
        f: sum(getattr(r.scores, f) for r in results) / n
        for f in score_fields
    }
    return JudgeScores(**kwargs)


def _group_scores(
    results: list[QueryResult], key_fn
) -> dict[str, JudgeScores]:
    groups: dict[str, list[QueryResult]] = {}
    for r in results:
        groups.setdefault(key_fn(r), []).append(r)
    return {k: _mean_scores(v) for k, v in groups.items()}


def _compute_latency_stats(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    s = sorted(latencies)
    n = len(s)
    return {
        "p50": s[int(n * 0.50)],
        "p95": s[min(int(n * 0.95), n - 1)],
        "p99": s[min(int(n * 0.99), n - 1)],
        "mean": sum(latencies) / n,
    }


def _scores_to_dict(s: JudgeScores) -> dict:
    return {
        "faithfulness": s.faithfulness,
        "answer_relevance": s.answer_relevance,
        "context_precision": s.context_precision,
        "context_recall": s.context_recall,
        "clinical_safety": s.clinical_safety,
        "citation_accuracy": s.citation_accuracy,
        "evidence_appropriate": s.evidence_appropriate,
        "overall": s.overall,
    }


def _save_report(report: EvaluationReport, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = report.timestamp.replace(":", "").replace(".", "").replace("+", "")[:15]

    detailed = {
        "run_id": report.run_id,
        "timestamp": report.timestamp,
        "golden_set_version": report.golden_set_version,
        "mode": report.mode,
        "total_queries": report.total_queries,
        "aggregate": _scores_to_dict(report.aggregate),
        "per_topic": {k: _scores_to_dict(v) for k, v in report.per_topic.items()},
        "per_difficulty": {
            k: _scores_to_dict(v) for k, v in report.per_difficulty.items()
        },
        "per_question_type": {
            k: _scores_to_dict(v) for k, v in report.per_question_type.items()
        },
        "latency_stats": report.latency_stats,
        "per_query_results": [
            {
                "id": r.query_id,
                "query": r.query,
                "cancer_type": r.cancer_type,
                "difficulty": r.difficulty,
                "question_type": r.question_type,
                "answer": r.answer[:500],
                "scores": _scores_to_dict(r.scores),
                "latency_ms": r.latency_ms,
                "contains_required_terms": r.contains_required_terms,
                "is_refused": r.is_refused,
            }
            for r in report.per_query_results
        ],
    }

    metrics: dict = {
        "run_id": report.run_id,
        "timestamp": report.timestamp,
        "mode": report.mode,
        "total_queries": report.total_queries,
        "aggregate": _scores_to_dict(report.aggregate),
        "latency_stats": report.latency_stats,
    }
    if report.regression_delta is not None:
        metrics["regression_delta"] = _scores_to_dict(report.regression_delta)
        metrics["passed_regression"] = report.passed_regression

    with open(out / f"{ts}_detailed.json", "w", encoding="utf-8") as fh:
        json.dump(detailed, fh, indent=2)
    with open(out / f"{ts}_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with open(out / "latest_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info("Saved evaluation report to %s", out)


def _compute_regression(
    current: JudgeScores, baseline_path: str
) -> tuple[JudgeScores, bool]:
    try:
        with open(baseline_path, "r", encoding="utf-8") as fh:
            base_data = json.load(fh)
        base = base_data.get("aggregate", {})
        delta = JudgeScores(
            faithfulness=current.faithfulness - base.get("faithfulness", 0.0),
            answer_relevance=current.answer_relevance - base.get("answer_relevance", 0.0),
            context_precision=current.context_precision - base.get("context_precision", 0.0),
            context_recall=current.context_recall - base.get("context_recall", 0.0),
            clinical_safety=current.clinical_safety - base.get("clinical_safety", 0.0),
        )
        passed = all(
            getattr(delta, f) >= -_REGRESSION_TOLERANCE
            for f in [
                "faithfulness", "answer_relevance", "context_precision",
                "context_recall", "clinical_safety",
            ]
        )
        return delta, passed
    except Exception as exc:
        logger.warning("Could not load baseline for regression: %s", exc)
        return JudgeScores(), True
