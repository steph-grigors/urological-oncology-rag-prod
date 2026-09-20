"""
Unit tests for CohereReranker ordering.

The class computes a weighted final score per chunk:

    final_score = relevance^0.70 * design_weight^0.15 * (recency * rct_boost)^0.15

and the module docstring presents that as the ranking formula. It was computed
and then thrown away: results were appended in the order Cohere returned them
and never re-sorted, so the study-design weight, the recency tiers and the RCT
landmark boost had no effect on anything. The generation layer received chunks
in pure Cohere relevance order, and [Doc 1] was always Cohere's top hit.

These tests pin the ordering contract. The Cohere client is a stub -- no
network, no key.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from src.db.vector_store import ScoredChunk
from src.retrieval.reranker import CohereReranker

THIS_YEAR = datetime.datetime.now().year


# ── Stub Cohere ──────────────────────────────────────────────────────────────

@dataclass
class _Result:
    index: int
    relevance_score: float


def _reranker_returning(scores: list[float]) -> CohereReranker:
    """A reranker whose Cohere call returns `scores` in descending order,
    which is how the real API responds."""
    reranker = CohereReranker(api_key="fake-key-not-used")
    client = MagicMock()
    ordered = sorted(enumerate(scores), key=lambda p: p[1], reverse=True)
    client.rerank.return_value = MagicMock(
        results=[_Result(index=i, relevance_score=s) for i, s in ordered]
    )
    reranker._client = client
    return reranker


def _chunk(chunk_id: str, *, design: str, year: int) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        score=0.0,
        metadata={"study_design": design, "year": year, "title": chunk_id},
    )


# ── Ordering ─────────────────────────────────────────────────────────────────

class TestResultsAreSortedByFinalScore:
    def test_output_is_descending_by_score(self):
        chunks = [
            _chunk("a", design="review", year=THIS_YEAR - 12),
            _chunk("b", design="rct", year=THIS_YEAR - 1),
            _chunk("c", design="cohort", year=THIS_YEAR - 6),
        ]
        ranked = _reranker_returning([0.80, 0.78, 0.60]).rerank("q", chunks, top_n=3)

        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True), (
            f"results not ordered by final_score: {scores}"
        )

    def test_recent_rct_overtakes_slightly_more_relevant_old_review(self):
        """The scenario the weighting exists for. A narrative review from 12
        years ago edges out a current randomised trial on raw relevance. The
        weights are meant to reverse that; before the sort they could not."""
        chunks = [
            _chunk("old_review", design="review", year=THIS_YEAR - 12),
            _chunk("recent_rct", design="rct", year=THIS_YEAR - 1),
        ]
        ranked = _reranker_returning([0.80, 0.78]).rerank("q", chunks, top_n=2)

        assert ranked[0].chunk_id == "recent_rct"
        assert ranked[1].chunk_id == "old_review"

    def test_strong_relevance_still_wins(self):
        """The weights are a nudge, not an override: a clearly more relevant
        older review must not be displaced by a marginal recent trial."""
        chunks = [
            _chunk("old_review", design="review", year=THIS_YEAR - 12),
            _chunk("recent_rct", design="rct", year=THIS_YEAR - 1),
        ]
        ranked = _reranker_returning([0.95, 0.40]).rerank("q", chunks, top_n=2)

        assert ranked[0].chunk_id == "old_review"


# ── What the sort must not disturb ───────────────────────────────────────────

class TestSortPreservesEverythingElse:
    def test_relevance_score_stays_attached_to_its_own_chunk(self):
        """retriever.retrieve grades and thresholds on relevance_score, and
        confidence is computed from it, so a reorder that mismatched chunk and
        score would corrupt both."""
        chunks = [
            _chunk("a", design="review", year=THIS_YEAR - 12),
            _chunk("b", design="rct", year=THIS_YEAR - 1),
            _chunk("c", design="cohort", year=THIS_YEAR - 6),
        ]
        expected = {"a": 0.80, "b": 0.78, "c": 0.60}
        ranked = _reranker_returning([0.80, 0.78, 0.60]).rerank("q", chunks, top_n=3)

        for r in ranked:
            assert r.relevance_score == pytest.approx(expected[r.chunk_id])
            assert r.text == f"text for {r.chunk_id}"
            assert r.metadata["title"] == r.chunk_id

    def test_no_chunk_is_gained_or_lost(self):
        chunks = [
            _chunk(name, design=d, year=y)
            for name, d, y in [("a", "review", THIS_YEAR - 12),
                               ("b", "rct", THIS_YEAR - 1),
                               ("c", "cohort", THIS_YEAR - 6)]
        ]
        ranked = _reranker_returning([0.80, 0.78, 0.60]).rerank("q", chunks, top_n=3)
        assert sorted(r.chunk_id for r in ranked) == ["a", "b", "c"]

    def test_passthrough_order_is_untouched(self):
        """With no API key the class must still return RRF order verbatim --
        tests/integration/test_retrieval_pipeline.py pins this too."""
        reranker = CohereReranker(api_key="")
        chunks = [
            ScoredChunk(chunk_id="first", text="t", score=0.9, metadata={}),
            ScoredChunk(chunk_id="second", text="t", score=0.5, metadata={}),
            ScoredChunk(chunk_id="third", text="t", score=0.1, metadata={}),
        ]
        ranked = reranker.rerank("q", chunks, top_n=3)
        assert [r.chunk_id for r in ranked] == ["first", "second", "third"]

    def test_api_failure_still_falls_back_to_rrf_order(self):
        reranker = CohereReranker(api_key="fake-key-not-used")
        client = MagicMock()
        client.rerank.side_effect = RuntimeError("cohere is down")
        reranker._client = client

        chunks = [
            ScoredChunk(chunk_id="first", text="t", score=0.9, metadata={}),
            ScoredChunk(chunk_id="second", text="t", score=0.5, metadata={}),
        ]
        ranked = reranker.rerank("q", chunks, top_n=2)
        assert [r.chunk_id for r in ranked] == ["first", "second"]

    def test_empty_input_returns_empty(self):
        assert _reranker_returning([]).rerank("q", [], top_n=5) == []


# ── Corpus reality check ─────────────────────────────────────────────────────

class TestUniformMetadataIsStillOrdered:
    """The production corpus currently carries study_design "unknown" for
    essentially every chunk, so the design weight is a constant. Recency still
    varies, and the sort must still hold."""

    def test_ordering_holds_when_design_is_uniformly_unknown(self):
        chunks = [
            _chunk("old", design="unknown", year=THIS_YEAR - 14),
            _chunk("new", design="unknown", year=THIS_YEAR - 1),
        ]
        ranked = _reranker_returning([0.70, 0.68]).rerank("q", chunks, top_n=2)

        assert ranked[0].chunk_id == "new"
        assert [r.score for r in ranked] == sorted([r.score for r in ranked], reverse=True)
