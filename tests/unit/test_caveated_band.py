"""
Tests that the caveated band exists in practice, not only in the enum.

ConfidenceGate has four states. Two of them were unreachable in combination:

  confidence was averaged over chunks that had already passed a
  >= CONFIDENCE_LOW filter, so it could not fall below CONFIDENCE_LOW whenever
  anything survived — which is the entire caveated band (0.2 to 0.45);

  and generator.generate collapsed everything that was not HIGH into the
  hedged prompt, so even a caveated score would have produced a hedged answer.

Measured on production before the fix: a penile sarcomatoid query reported
0.482 (hedged) where scoring the full candidate set gives 0.310 (caveated).
Four of its five chunks were irrelevant.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config.constants import CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_REFUSE
from src.generation.confidence import ConfidenceGate, gate
from src.generation.prompts import (
    CAVEATED_ANSWER_PREFIX,
    HEDGED_ANSWER_PREFIX,
    build_prompt,
)


class TestEachBandGetsItsOwnPrompt:
    def _prefix_for(self, level: str) -> str:
        content = build_prompt("q", [], confidence_level=level)[0]["content"]
        return content

    def test_high_gets_no_prefix(self):
        assert not self._prefix_for("high").startswith(("**Note:**", "> "))

    def test_hedged_and_caveated_differ(self):
        """They produced identical prompts before."""
        assert self._prefix_for("hedged") != self._prefix_for("caveated")

    def test_caveated_is_the_stronger_warning(self):
        caveated = self._prefix_for("caveated")
        assert CAVEATED_ANSWER_PREFIX in caveated
        assert "do not directly address it" in caveated
        assert "does and does not cover" in caveated

    def test_hedged_is_unchanged(self):
        assert HEDGED_ANSWER_PREFIX in self._prefix_for("hedged")

    def test_unknown_level_fails_cautious(self):
        assert HEDGED_ANSWER_PREFIX in self._prefix_for("something-else")


class TestGateBandsAreContiguous:
    @pytest.mark.parametrize("score,expected", [
        (0.95, ConfidenceGate.HIGH),
        (CONFIDENCE_HIGH, ConfidenceGate.HIGH),
        (0.60, ConfidenceGate.HEDGED),
        (CONFIDENCE_LOW, ConfidenceGate.HEDGED),
        (0.31, ConfidenceGate.CAVEATED),
        (CONFIDENCE_REFUSE, ConfidenceGate.CAVEATED),
        (0.10, ConfidenceGate.REFUSED),
        (0.0, ConfidenceGate.REFUSED),
    ])
    def test_band(self, score, expected):
        assert gate(score) == expected

    def test_the_measured_production_value_is_caveated(self):
        """0.310 is what the penile sarcomatoid query scores over its full
        candidate set. It reported 0.482 before, which is hedged."""
        assert gate(0.310) == ConfidenceGate.CAVEATED
        assert gate(0.482) == ConfidenceGate.HEDGED


class TestGeneratorUsesTheRetrieverScore:
    def _generate(self, chunks, confidence_score):
        from src.generation.generator import ClinicalGenerator
        from src.retrieval.reranker import RankedChunk

        llm = MagicMock()
        llm.provider = "anthropic"
        llm.complete.return_value = MagicMock(
            content="Answer [Doc 1].", input_tokens=1, output_tokens=1, model="m")
        with patch("src.generation.generator.apply_regulatory_warnings",
                   side_effect=lambda a, **k: a):
            result = ClinicalGenerator(llm_client=llm).generate(
                "q", chunks, confidence_score=confidence_score)
        return result, llm.complete.call_args[0][1][0]["content"]

    def _chunk(self, score):
        from src.retrieval.reranker import RankedChunk

        return RankedChunk(chunk_id="c1", text="t", score=score,
                           relevance_score=score,
                           metadata={"pmcid": "1", "evidence_level": 2})

    def test_a_caveated_score_produces_the_caveated_prompt(self):
        """The chunk itself scores 0.9; the retriever's score says retrieval
        went badly overall. The retriever's view must win."""
        result, prompt = self._generate([self._chunk(0.9)], confidence_score=0.31)
        assert result.evidence_quality == "caveated"
        assert CAVEATED_ANSWER_PREFIX in prompt

    def test_a_high_score_produces_no_prefix(self):
        result, prompt = self._generate([self._chunk(0.9)], confidence_score=0.95)
        assert result.evidence_quality == "high"
        assert CAVEATED_ANSWER_PREFIX not in prompt
        assert HEDGED_ANSWER_PREFIX not in prompt

    def test_the_reported_score_is_the_retriever_score(self):
        result, _ = self._generate([self._chunk(0.9)], confidence_score=0.31)
        assert result.confidence_score == pytest.approx(0.31)

    def test_omitting_it_still_works(self):
        """Backwards compatible: callers that pass nothing get the old
        self-computed behaviour."""
        result, _ = self._generate([self._chunk(0.9)], confidence_score=None)
        assert 0.0 <= result.confidence_score <= 1.0
