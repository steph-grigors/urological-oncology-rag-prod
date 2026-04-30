"""
Integration tests for the full retrieval pipeline.

Requires:
    - Qdrant running (docker-compose up qdrant)
    - Postgres running (docker-compose up postgres)
    - OPENAI_API_KEY set (real embedding call)
    - A small fixture corpus indexed (5 sample papers from tests/fixtures/)

Marked with @pytest.mark.integration — excluded from default test run.

Tests cover:
    - VectorSearch returns non-empty results for a known query.
    - BM25Search returns non-empty results for an exact drug name query.
    - HybridSearch returns merged, deduplicated results.
    - Reranker reorders results: if COHERE_API_KEY set, top result changes
      vs. fusion-only order for at least one query in the fixture set.
    - Reranker degrades gracefully when COHERE_API_KEY is empty.
    - Topic filter: prostate-only filter returns no kidney chunks.
    - Year filter: year_min=2020 returns no pre-2020 chunks.
    - End-to-end Retriever.retrieve() returns a RetrievalResult with correct
      field types and non-empty chunks list.
    - Latency: retrieve() completes in < 5s (with real embedding API).
"""

import pytest


pytestmark = pytest.mark.integration


class TestVectorSearch:
    def test_returns_results_for_known_query(self):
        raise NotImplementedError

    def test_topic_filter(self):
        raise NotImplementedError

    def test_year_filter(self):
        raise NotImplementedError


class TestBM25Search:
    def test_exact_drug_name_match(self):
        raise NotImplementedError


class TestHybridSearch:
    def test_merged_results_deduplicated(self):
        raise NotImplementedError


class TestReranker:
    def test_reranks_when_cohere_available(self):
        raise NotImplementedError

    def test_graceful_degradation_without_cohere(self):
        raise NotImplementedError


class TestRetriever:
    def test_end_to_end_returns_retrieval_result(self):
        raise NotImplementedError

    def test_latency_under_five_seconds(self):
        raise NotImplementedError
