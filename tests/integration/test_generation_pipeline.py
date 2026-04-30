"""
Integration tests for the generation pipeline.

Requires OPENAI_API_KEY or ANTHROPIC_API_KEY (real LLM calls).
Marked with @pytest.mark.integration.

Tests cover:
    - Generator.generate() returns a non-empty answer string.
    - Answer contains at least one [Doc N] citation.
    - Hallucinated citations (doc numbers beyond num_docs) are stripped.
    - Anthropic provider: cache_hit field is True on repeated identical call.
    - OpenAI provider: result has correct provider field.
    - Confidence gate=REFUSED → answer is the refusal text, not empty string.
    - Confidence gate=HEDGED → answer starts with HEDGED_ANSWER_PREFIX.
    - Medical disclaimer is appended to all non-refused answers.
    - Streaming mode yields at least one chunk before completing.
    - End-to-end query through Retriever → Generator → AuditLogger:
        audit record is written to Postgres with correct query_id.
"""

import pytest


pytestmark = pytest.mark.integration


class TestGeneratorOutput:
    def test_returns_non_empty_answer(self):
        raise NotImplementedError

    def test_answer_contains_citation(self):
        raise NotImplementedError

    def test_hallucinated_citations_stripped(self):
        raise NotImplementedError

    def test_disclaimer_appended(self):
        raise NotImplementedError


class TestProviders:
    def test_anthropic_cache_hit_on_repeat(self):
        raise NotImplementedError

    def test_openai_provider_field(self):
        raise NotImplementedError


class TestConfidenceGating:
    def test_refused_gate_returns_refusal_text(self):
        raise NotImplementedError

    def test_hedged_gate_prefix(self):
        raise NotImplementedError


class TestStreaming:
    def test_yields_chunks_before_completing(self):
        raise NotImplementedError


class TestEndToEnd:
    def test_audit_record_written(self):
        raise NotImplementedError
