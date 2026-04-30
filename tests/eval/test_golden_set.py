"""
Golden set evaluation tests — assert system quality meets minimum thresholds.

These tests run the full RAG pipeline against the golden query set and fail
if any metric drops below the defined floor.  They are the regression gate
before merging pipeline changes.

Marked with @pytest.mark.eval — excluded from unit and integration runs.
Run with: pytest tests/eval -m eval

Quality floors (to be tuned as the system matures):
    avg_faithfulness        >= 0.90
    avg_answer_relevance    >= 0.85
    avg_context_precision   >= 0.80
    avg_clinical_safety     >= 0.99   # near-zero tolerance for unsafe answers
    p95_latency_ms          <= 8000

Per-topic floors (all topics must pass individually):
    avg_quality_per_topic   >= 0.80

Tests:
    test_overall_faithfulness_above_floor
    test_overall_relevance_above_floor
    test_overall_context_precision_above_floor
    test_clinical_safety_above_floor
    test_p95_latency_below_ceiling
    test_per_topic_quality_above_floor[prostate]
    test_per_topic_quality_above_floor[bladder]
    test_per_topic_quality_above_floor[kidney]
    test_per_topic_quality_above_floor[testicular]
    test_no_refused_answers_on_easy_queries
"""

import pytest


pytestmark = pytest.mark.eval

FAITHFULNESS_FLOOR = 0.90
RELEVANCE_FLOOR = 0.85
CONTEXT_PRECISION_FLOOR = 0.80
CLINICAL_SAFETY_FLOOR = 0.99
P95_LATENCY_MS_CEILING = 8000
PER_TOPIC_QUALITY_FLOOR = 0.80


class TestOverallMetrics:
    def test_faithfulness_above_floor(self):
        raise NotImplementedError

    def test_relevance_above_floor(self):
        raise NotImplementedError

    def test_context_precision_above_floor(self):
        raise NotImplementedError

    def test_clinical_safety_above_floor(self):
        raise NotImplementedError

    def test_p95_latency_below_ceiling(self):
        raise NotImplementedError


class TestPerTopicMetrics:
    @pytest.mark.parametrize("topic", ["prostate", "bladder", "kidney", "testicular"])
    def test_topic_quality_above_floor(self, topic: str):
        raise NotImplementedError


class TestEdgeCases:
    def test_no_refusals_on_easy_queries(self):
        raise NotImplementedError
