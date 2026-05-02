"""
Golden set evaluation tests — assert system quality meets minimum thresholds.

These tests run the full RAG pipeline against the golden query set and fail
if any metric drops below the defined floor.  They are the regression gate
before merging pipeline changes.

Marked with @pytest.mark.eval — excluded from unit and integration runs.
Run with: pytest tests/eval -m eval

Quality floors:
    avg_faithfulness        >= 0.90
    avg_answer_relevance    >= 0.85
    avg_context_precision   >= 0.80
    avg_clinical_safety     >= 0.99
    p95_latency_ms          <= 8000
    avg_quality_per_topic   >= 0.80
"""

import pytest

from src.evaluation.runner import EvaluationReport, run_evaluation
from src.retrieval.retriever import RetrievalResult
from src.retrieval.reranker import RankedChunk

pytestmark = pytest.mark.eval

FAITHFULNESS_FLOOR = 0.90
RELEVANCE_FLOOR = 0.85
CONTEXT_PRECISION_FLOOR = 0.80
CLINICAL_SAFETY_FLOOR = 0.99
P95_LATENCY_MS_CEILING = 8000
PER_TOPIC_QUALITY_FLOOR = 0.80


# ── Mock pipeline ─────────────────────────────────────────────────────────────

def _make_mock_retriever():
    class _MockRetriever:
        def retrieve(self, query, **kwargs):
            chunks = [
                RankedChunk(
                    chunk_id=f"chunk_{i}",
                    text=(
                        f"{query} treatment randomized controlled trial "
                        "evidence suggests improvement in outcomes"
                    ),
                    score=0.88,
                    relevance_score=0.88,
                    metadata={
                        "evidence_level": 2,
                        "study_design": "rct",
                        "cancer_type": ["prostate"],
                    },
                )
                for i in range(5)
            ]
            return RetrievalResult(
                query=query,
                chunks=chunks,
                retrieval_confidence=0.88,
                num_candidates=20,
            )

    return _MockRetriever()


def _make_mock_generator():
    from src.generation.generator import GenerationResult

    class _MockGenerator:
        def generate(self, query, ranked_chunks, conversation_history=None):
            return GenerationResult(
                answer=(
                    f"Based on randomized controlled trials [Doc 1], "
                    f"{query} [Doc 2]. "
                    "Evidence suggests improvement in clinical outcomes [Doc 3]."
                ),
                citations=[1, 2, 3],
                evidence_quality="high",
                model_used="mock",
                provider="mock",
                prompt_tokens=100,
                completion_tokens=50,
                confidence_score=0.88,
            )

    return _MockGenerator()


# ── Module-scoped fixture (runs the pipeline once for all tests) ──────────────

@pytest.fixture(scope="module")
def eval_report() -> EvaluationReport:
    return run_evaluation(
        retriever=_make_mock_retriever(),
        golden_set_path="tests/fixtures/golden_queries.json",
        output_dir=None,
        mode="full",
        generator=_make_mock_generator(),
    )


# ── TestOverallMetrics ────────────────────────────────────────────────────────

class TestOverallMetrics:
    def test_faithfulness_above_floor(self, eval_report):
        score = eval_report.aggregate.faithfulness
        assert score >= FAITHFULNESS_FLOOR, (
            f"Faithfulness {score:.3f} below floor {FAITHFULNESS_FLOOR}"
        )

    def test_relevance_above_floor(self, eval_report):
        score = eval_report.aggregate.answer_relevance
        assert score >= RELEVANCE_FLOOR, (
            f"Answer relevance {score:.3f} below floor {RELEVANCE_FLOOR}"
        )

    def test_context_precision_above_floor(self, eval_report):
        score = eval_report.aggregate.context_precision
        assert score >= CONTEXT_PRECISION_FLOOR, (
            f"Context precision {score:.3f} below floor {CONTEXT_PRECISION_FLOOR}"
        )

    def test_clinical_safety_above_floor(self, eval_report):
        score = eval_report.aggregate.clinical_safety
        assert score >= CLINICAL_SAFETY_FLOOR, (
            f"Clinical safety {score:.3f} below floor {CLINICAL_SAFETY_FLOOR}"
        )

    def test_p95_latency_below_ceiling(self, eval_report):
        p95 = eval_report.latency_stats["p95"]
        assert p95 <= P95_LATENCY_MS_CEILING, (
            f"p95 latency {p95:.1f}ms above ceiling {P95_LATENCY_MS_CEILING}ms"
        )


# ── TestPerTopicMetrics ───────────────────────────────────────────────────────

class TestPerTopicMetrics:
    @pytest.mark.parametrize(
        "topic", ["prostate", "bladder", "kidney", "testicular"]
    )
    def test_topic_quality_above_floor(self, eval_report, topic):
        topic_scores = eval_report.per_topic.get(topic)
        assert topic_scores is not None, f"No results found for topic {topic!r}"
        quality = topic_scores.overall
        assert quality >= PER_TOPIC_QUALITY_FLOOR, (
            f"Topic {topic!r} overall quality {quality:.3f} "
            f"below floor {PER_TOPIC_QUALITY_FLOOR}"
        )


# ── TestEdgeCases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_no_refusals_on_easy_queries(self, eval_report):
        refused = [
            r.query_id
            for r in eval_report.per_query_results
            if r.difficulty == "easy" and r.is_refused
        ]
        assert not refused, f"Refused answers on easy queries: {refused}"
