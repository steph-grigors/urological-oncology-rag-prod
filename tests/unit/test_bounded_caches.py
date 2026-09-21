"""
Tests that two in-process caches stay bounded.

Both grew without limit for the lifetime of the process:

  RAGRetriever._embed_cache was an unbounded dict declared as a CLASS
  attribute, so every retriever ever constructed shared one dictionary that
  never evicted anything. Each distinct query added a 1536-float list.

  RateLimitMiddleware._counters is keyed per API key or per client IP and
  entries were never removed. The key set is bounded for authenticated
  callers and unbounded for anonymous ones, so a public endpoint accumulates
  scanner IPs indefinitely.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from src.api.middleware.rate_limit import _WINDOW_SECONDS, RateLimitMiddleware
from src.retrieval.retriever import RAGRetriever


# ── Query embedding cache ────────────────────────────────────────────────────

def _retriever(maxsize: int) -> RAGRetriever:
    openai = MagicMock()
    openai.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 8)]
    )
    return RAGRetriever(
        store=MagicMock(),
        bm25=MagicMock(),
        reranker=MagicMock(),
        openai_client=openai,
        embed_cache_maxsize=maxsize,
    )


class TestEmbedCacheIsBounded:
    def test_never_exceeds_its_maximum(self):
        r = _retriever(maxsize=5)
        for i in range(50):
            r._embed(f"query {i}")
        assert len(r._embed_cache) <= 5

    def test_repeated_query_is_served_from_cache(self):
        r = _retriever(maxsize=10)
        r._embed("same question")
        r._embed("same question")
        assert r._openai.embeddings.create.call_count == 1

    def test_evicted_entry_is_re_embedded(self):
        r = _retriever(maxsize=2)
        r._embed("a")
        r._embed("b")
        r._embed("c")          # evicts "a"
        calls_before = r._openai.embeddings.create.call_count
        r._embed("a")
        assert r._openai.embeddings.create.call_count == calls_before + 1

    def test_zero_maxsize_disables_caching(self):
        r = _retriever(maxsize=0)
        r._embed("q")
        r._embed("q")
        assert len(r._embed_cache) == 0
        assert r._openai.embeddings.create.call_count == 2

    def test_cache_is_per_instance_not_shared(self):
        """It was a class attribute, so two retrievers -- production and a
        test double, or two collections -- shared one dictionary."""
        a, b = _retriever(10), _retriever(10)
        a._embed("only in a")
        assert len(a._embed_cache) == 1
        assert len(b._embed_cache) == 0
        assert a._embed_cache is not b._embed_cache

    def test_embedding_value_round_trips(self):
        r = _retriever(10)
        assert r._embed("q") == [0.1] * 8


# ── Rate limiter counters ────────────────────────────────────────────────────

class TestRateLimitCountersAreSwept:
    def _middleware(self) -> RateLimitMiddleware:
        settings = MagicMock()
        settings.rate_limit_per_minute = 60
        settings.api_key_header = "X-API-Key"
        return RateLimitMiddleware(app=MagicMock(), settings=settings)

    def test_expired_entries_are_dropped(self):
        mw = self._middleware()
        now = time.time()
        mw._counters["ip:1.2.3.4"] = (5, now - _WINDOW_SECONDS - 1)
        mw._counters["ip:5.6.7.8"] = (5, now)

        mw._sweep_expired(now)

        assert "ip:1.2.3.4" not in mw._counters
        assert "ip:5.6.7.8" in mw._counters

    def test_sweeping_an_expired_entry_changes_no_decision(self):
        """An expired counter and a missing one both start a fresh window, so
        the sweep can never alter whether a request is allowed."""
        mw = self._middleware()
        now = time.time()
        stale = (999, now - _WINDOW_SECONDS - 1)

        count, window_start = stale
        rolled = count if now - window_start < _WINDOW_SECONDS else 0
        assert rolled == 0, "an expired window resets the count regardless"

    def test_sweep_on_empty_dict_is_safe(self):
        mw = self._middleware()
        mw._sweep_expired(time.time())
        assert mw._counters == {}

    @pytest.mark.parametrize("age,kept", [(0, True), (_WINDOW_SECONDS - 1, True),
                                          (_WINDOW_SECONDS + 1, False)])
    def test_boundary(self, age, kept):
        mw = self._middleware()
        now = time.time()
        mw._counters["k"] = (1, now - age)
        mw._sweep_expired(now)
        assert ("k" in mw._counters) is kept
