"""
Unit tests for src/generation/confidence.py.

Tests operate on synthetic RetrievalResult objects — no LLM or database
calls.

Tests cover:
    gate() returns REFUSED when confidence < CONFIDENCE_REFUSE.
    gate() returns CAVEATED when CONFIDENCE_REFUSE <= c < CONFIDENCE_LOW.
    gate() returns HEDGED when CONFIDENCE_LOW <= c < CONFIDENCE_HIGH.
    gate() returns HIGH when confidence >= CONFIDENCE_HIGH.

    compute_confidence():
        - All chunks with score 1.0 → confidence near 1.0.
        - All chunks with score 0.0 → confidence near 0.0.
        - Single-paper retrieval (low diversity) → penalised confidence.
        - Topic mismatch (all kidney chunks for prostate query) → penalised.
        - High score spread → lower confidence than uniform high scores.

    confidence_to_metadata() includes all sub-score keys.
    Thresholds from settings override module-level constants.
"""

import pytest


# TODO: import compute_confidence, gate, ConfidenceGate from src.generation.confidence


class TestGateDecisions:
    def test_refused_below_threshold(self):
        raise NotImplementedError

    def test_caveated_band(self):
        raise NotImplementedError

    def test_hedged_band(self):
        raise NotImplementedError

    def test_high_above_threshold(self):
        raise NotImplementedError


class TestComputeConfidence:
    def test_perfect_scores_high_confidence(self):
        raise NotImplementedError

    def test_zero_scores_low_confidence(self):
        raise NotImplementedError

    def test_single_paper_penalty(self):
        raise NotImplementedError

    def test_topic_mismatch_penalty(self):
        raise NotImplementedError

    def test_high_spread_penalty(self):
        raise NotImplementedError


class TestMetadata:
    def test_confidence_metadata_has_required_keys(self):
        raise NotImplementedError
