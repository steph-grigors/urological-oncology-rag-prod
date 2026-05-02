"""
Unit tests for src/ingestion/extract_metadata.py.

All LLM calls are mocked — these tests verify prompt construction,
response parsing, and failure handling without touching the OpenAI API.
"""

import json
from unittest.mock import MagicMock

import pytest

from src.ingestion.extract_metadata import (
    STUDY_DESIGN_OPTIONS,
    ExtractionResult,
    extract_metadata,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

_VALID_JSON = {
    "study_design": "rct",
    "sample_size": 500,
    "primary_outcome": "Overall survival at 5 years",
}


def _mock_client(response_json: dict) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = json.dumps(response_json)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create.return_value = resp
    return client


# ── TestSuccessfulExtraction ──────────────────────────────────────────────────

class TestSuccessfulExtraction:
    def test_returns_all_fields(self, tmp_path):
        client = _mock_client(_VALID_JSON)
        result = extract_metadata(
            "PMID001",
            "A randomized trial of 500 patients with prostate cancer.",
            client,
            cache_path=str(tmp_path / "cache.json"),
        )
        assert result.pmid == "PMID001"
        assert result.study_design == "rct"
        assert result.sample_size == 500
        assert result.primary_outcome == "Overall survival at 5 years"
        assert result.extraction_failed is False

    def test_study_design_normalised(self, tmp_path):
        client = _mock_client(_VALID_JSON)
        result = extract_metadata(
            "PMID002",
            "A randomized trial.",
            client,
            cache_path=str(tmp_path / "cache.json"),
        )
        assert result.study_design in STUDY_DESIGN_OPTIONS

    def test_prompt_contains_abstract(self, tmp_path):
        abstract = "This is a prospective cohort study of 200 patients with bladder cancer."
        client = _mock_client(_VALID_JSON)
        extract_metadata(
            "PMID003",
            abstract,
            client,
            cache_path=str(tmp_path / "cache.json"),
        )
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert abstract in user_content

    def test_prompt_excludes_full_body(self, tmp_path):
        short = "RCT of prostate cancer treatment."
        # Build a string longer than 2000 chars with a unique overflow marker
        padding = "x" * (2000 - len(short))
        overflow_marker = "OVERFLOW_UNIQUE_MARKER_ZZZ"
        long_text = short + padding + overflow_marker
        assert len(long_text) > 2000

        client = _mock_client(_VALID_JSON)
        extract_metadata(
            "PMID004",
            long_text,
            client,
            cache_path=str(tmp_path / "cache.json"),
        )
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert overflow_marker not in user_content


# ── TestFailureHandling ───────────────────────────────────────────────────────

class TestFailureHandling:
    def test_json_parse_error_sets_failed_flag(self, tmp_path):
        client = MagicMock()
        msg = MagicMock()
        msg.content = "NOT VALID JSON {{{"
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        client.chat.completions.create.return_value = resp

        result = extract_metadata(
            "PMID005",
            "Some abstract.",
            client,
            cache_path=str(tmp_path / "cache.json"),
        )
        assert result.extraction_failed is True

    def test_api_timeout_sets_failed_flag(self, tmp_path):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("Connection timeout")

        result = extract_metadata(
            "PMID006",
            "Some abstract.",
            client,
            cache_path=str(tmp_path / "cache.json"),
        )
        assert result.extraction_failed is True

    def test_failed_extraction_does_not_raise(self, tmp_path):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API error")

        result = extract_metadata(
            "PMID007",
            "Some abstract.",
            client,
            cache_path=str(tmp_path / "cache.json"),
        )
        assert isinstance(result, ExtractionResult)
        assert result.extraction_failed is True


# ── TestCaching ───────────────────────────────────────────────────────────────

class TestCaching:
    def test_second_call_skips_openai(self, tmp_path):
        client = _mock_client(_VALID_JSON)
        cache_path = str(tmp_path / "cache.json")

        extract_metadata("PMID008", "Some abstract.", client, cache_path=cache_path)
        extract_metadata("PMID008", "Some abstract.", client, cache_path=cache_path)

        assert client.chat.completions.create.call_count == 1
