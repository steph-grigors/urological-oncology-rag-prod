"""
Unit tests for src/ingestion/extract_metadata.py.

All LLM calls are mocked — these tests verify prompt construction,
response parsing, and failure handling without touching the OpenAI API.

Tests cover:
    - Successful extraction returns a fully-populated MetadataExtractionResult.
    - study_design value is normalised to a value in STUDY_DESIGN_HIERARCHY.
    - OpenAI JSON parse error → extraction_failed=True, all fields None.
    - OpenAI API timeout → extraction_failed=True (no exception raised to caller).
    - Result is cached by pmc_id: second call does not invoke OpenAI.
    - Prompt sent to OpenAI contains the abstract text.
    - Prompt does NOT contain full body text (cost control).
    - cancer_subtype is None when the paper covers the full topic broadly.
"""

import pytest


# TODO: import extract_metadata, MetadataExtractionResult from src.ingestion.extract_metadata


class TestSuccessfulExtraction:
    def test_returns_all_fields(self):
        raise NotImplementedError

    def test_study_design_normalised(self):
        raise NotImplementedError

    def test_prompt_contains_abstract(self):
        raise NotImplementedError

    def test_prompt_excludes_full_body(self):
        raise NotImplementedError


class TestFailureHandling:
    def test_json_parse_error_sets_failed_flag(self):
        raise NotImplementedError

    def test_api_timeout_sets_failed_flag(self):
        raise NotImplementedError

    def test_failed_extraction_does_not_raise(self):
        raise NotImplementedError


class TestCaching:
    def test_second_call_skips_openai(self):
        raise NotImplementedError
