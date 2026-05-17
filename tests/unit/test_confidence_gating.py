"""
Unit tests for src/generation/confidence.py, src/generation/llm_client.py,
and src/generation/generator.py (hallucination check).

No live API calls — SDK calls are mocked via pytest-mock.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.generation.confidence import (
    ConfidenceGate,
    compute_confidence,
    confidence_to_metadata,
    gate,
)
from src.retrieval.reranker import RankedChunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk(
    chunk_id: str,
    score: float,
    evidence_level: int = 3,
    cancer_type: list[str] | None = None,
    pmcid: str = "1001",
) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        text=f"Clinical evidence text for {chunk_id}.",
        score=score,
        relevance_score=score,
        metadata={
            "evidence_level": evidence_level,
            "cancer_type": cancer_type or ["prostate"],
            "pmcid": pmcid,
        },
    )


# ── Gate boundary decisions ───────────────────────────────────────────────────

class TestGateDecisions:
    def test_refused_below_threshold(self):
        assert gate(0.1) == ConfidenceGate.REFUSED

    def test_caveated_band(self):
        # CONFIDENCE_REFUSE=0.2 ≤ 0.3 < CONFIDENCE_LOW=0.45
        assert gate(0.3) == ConfidenceGate.CAVEATED

    def test_hedged_band(self):
        # CONFIDENCE_LOW=0.45 ≤ 0.6 < CONFIDENCE_HIGH=0.75
        assert gate(0.6) == ConfidenceGate.HEDGED

    def test_high_above_threshold(self):
        # 0.8 ≥ CONFIDENCE_HIGH=0.75
        assert gate(0.8) == ConfidenceGate.HIGH


# ── compute_confidence scoring ────────────────────────────────────────────────

class TestComputeConfidence:
    def test_perfect_scores_high_confidence(self):
        # High-quality evidence (level 1), diverse sources, perfect scores
        chunks = [
            _chunk(f"c{i}", 1.0, evidence_level=1, pmcid=str(i + 1))
            for i in range(5)
        ]
        result = compute_confidence(chunks)
        assert result.score >= 0.75
        assert result.sufficient is True

    def test_zero_scores_low_confidence(self):
        chunks = [_chunk(f"c{i}", 0.0) for i in range(5)]
        result = compute_confidence(chunks)
        assert result.score < 0.45
        assert result.sufficient is False

    def test_single_paper_penalty(self):
        # All chunks from same paper → penalised vs diverse sources
        single = [_chunk(f"c{i}", 0.6, pmcid="1001") for i in range(3)]
        diverse = [_chunk(f"c{i}", 0.6, pmcid=str(i + 1)) for i in range(3)]
        assert compute_confidence(single).score < compute_confidence(diverse).score

    def test_topic_mismatch_penalty(self):
        # All kidney chunks, prostate query → lower than matching query
        chunks = [
            _chunk(f"c{i}", 0.7, cancer_type=["kidney"], pmcid=str(i + 1))
            for i in range(3)
        ]
        mismatch = compute_confidence(chunks, query_cancer_types=["prostate"])
        match_ = compute_confidence(chunks, query_cancer_types=["kidney"])
        assert mismatch.score < match_.score

    def test_high_spread_penalty(self):
        # [0.7, 0.7, 0.7, 0.0, 0.0]: std ≈ 0.34 > 0.3 → penalty applied
        high_spread = [
            _chunk("c0", 0.7, pmcid="1"),
            _chunk("c1", 0.7, pmcid="2"),
            _chunk("c2", 0.7, pmcid="3"),
            _chunk("c3", 0.0, pmcid="4"),
            _chunk("c4", 0.0, pmcid="5"),
        ]
        uniform = [_chunk(f"c{i}", 0.7, pmcid=str(i + 1)) for i in range(5)]
        assert compute_confidence(high_spread).score < compute_confidence(uniform).score


# ── Metadata keys ─────────────────────────────────────────────────────────────

class TestMetadata:
    def test_confidence_metadata_has_required_keys(self):
        result = compute_confidence([_chunk("c0", 0.6, pmcid="1")])
        meta = confidence_to_metadata(result)
        for key in ("score", "gate", "sufficient", "reason"):
            assert key in meta, f"Missing key: {key}"


# ── LLMClient ─────────────────────────────────────────────────────────────────

class TestLLMClient:
    def test_unknown_provider_raises_configuration_error(self):
        from src.generation.llm_client import ConfigurationError, LLMClient
        with pytest.raises(ConfigurationError):
            LLMClient(provider="unknown_provider", model="some-model", api_key="key")

    def test_anthropic_provider_calls_sdk(self, mocker):
        mock_cls = mocker.patch("anthropic.Anthropic")
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="Answer text")]
        mock_resp.usage.input_tokens = 100
        mock_resp.usage.output_tokens = 50
        mock_client.messages.create.return_value = mock_resp

        from src.generation.llm_client import LLMClient
        client = LLMClient(provider="anthropic", model="claude-3-haiku-20240307", api_key="test")
        result = client.complete("System", [{"role": "user", "content": "Q"}])

        assert result.content == "Answer text"
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    def test_openai_provider_calls_sdk(self, mocker):
        mock_cls = mocker.patch("openai.OpenAI")
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "GPT answer"
        mock_resp.usage.prompt_tokens = 80
        mock_resp.usage.completion_tokens = 40
        mock_client.chat.completions.create.return_value = mock_resp

        from src.generation.llm_client import LLMClient
        client = LLMClient(provider="openai", model="gpt-4o-mini", api_key="test")
        result = client.complete("System", [{"role": "user", "content": "Q"}])

        assert result.content == "GPT answer"
        assert result.input_tokens == 80
        assert result.output_tokens == 40


# ── ClinicalGenerator — hallucination check ───────────────────────────────────

class TestClinicalGenerator:
    def _make_mock_llm(self, text: str) -> MagicMock:
        from src.generation.llm_client import LLMResponse
        mock_llm = MagicMock()
        mock_llm.provider = "anthropic"
        mock_llm.complete.return_value = LLMResponse(
            content=text,
            input_tokens=100,
            output_tokens=50,
            model="claude-3-haiku-20240307",
        )
        return mock_llm

    def test_hallucinated_citations_flagged(self):
        from src.generation.generator import ClinicalGenerator
        chunks = [_chunk(f"c{i}", 0.8, pmcid=str(i + 1)) for i in range(3)]
        gen = ClinicalGenerator(
            llm_client=self._make_mock_llm(
                "Enzalutamide improves survival [Doc 1]. See also [Doc 9]."
            )
        )
        result = gen.generate("Efficacy of enzalutamide?", chunks)
        assert 9 in result.hallucinated_citations
        assert "WARNING" in result.answer

    def test_valid_citations_not_flagged(self):
        from src.generation.generator import ClinicalGenerator
        chunks = [_chunk(f"c{i}", 0.8, pmcid=str(i + 1)) for i in range(3)]
        gen = ClinicalGenerator(
            llm_client=self._make_mock_llm(
                "Enzalutamide improves OS [Doc 1] and PFS [Doc 2]."
            )
        )
        result = gen.generate("Efficacy of enzalutamide?", chunks)
        assert result.hallucinated_citations == []
        assert "WARNING" not in result.answer
        assert 1 in result.citations
        assert 2 in result.citations

    def test_low_confidence_uses_fallback_llm(self):
        # REFUSED gate calls LLM with fallback prompt and prepends FALLBACK_DISCLAIMER.
        from src.generation.generator import ClinicalGenerator
        from src.generation.prompts import FALLBACK_DISCLAIMER
        chunks = [_chunk(f"c{i}", 0.0) for i in range(3)]
        gen = ClinicalGenerator(llm_client=self._make_mock_llm("General oncology knowledge answer."))
        result = gen.generate("Efficacy?", chunks)
        gen._llm.complete.assert_called_once()
        assert FALLBACK_DISCLAIMER.strip() in result.answer
