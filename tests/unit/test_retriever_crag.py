"""
Unit tests for cRAG-lite grading + PubMed fallback in RAGRetriever.retrieve().

The store/bm25/reranker collaborators are mocked so these tests exercise only
the grading/fallback branch added in src/retrieval/retriever.py, not the real
hybrid search pipeline (covered by tests/integration/test_retrieval_pipeline.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from config.constants import CONFIDENCE_LOW
from src.retrieval.reranker import RankedChunk
from src.retrieval.retriever import RAGRetriever


def _make_chunk(chunk_id: str, relevance_score: float) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        score=relevance_score,
        relevance_score=relevance_score,
        metadata={"study_design": "rct"},
    )


def _build_retriever(reranked: list[RankedChunk], web_fallback=None) -> RAGRetriever:
    store = MagicMock()
    store.search_dense.return_value = []
    bm25 = MagicMock()
    bm25.search.return_value = []
    reranker = MagicMock()
    reranker.rerank.return_value = reranked
    openai_client = MagicMock()
    openai_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.0] * 1536)]
    )

    retriever = RAGRetriever(
        store=store,
        bm25=bm25,
        reranker=reranker,
        openai_client=openai_client,
        web_fallback=web_fallback,
    )
    # Avoid cross-test pollution of the shared embed cache.
    retriever._embed_cache = {}
    return retriever


class TestChunkGrading:
    def test_incorrect_chunks_are_discarded(self):
        chunks = [
            _make_chunk("good", 0.9),
            _make_chunk("bad", CONFIDENCE_LOW - 0.1),
        ]
        retriever = _build_retriever(chunks)
        result = retriever.retrieve("does enzalutamide improve survival")
        assert [c.chunk_id for c in result.chunks] == ["good"]

    def test_ambiguous_chunks_are_kept(self):
        chunks = [_make_chunk("ambiguous", CONFIDENCE_LOW)]
        retriever = _build_retriever(chunks)
        result = retriever.retrieve("query")
        assert [c.chunk_id for c in result.chunks] == ["ambiguous"]

    def test_bad_chunk_no_longer_dilutes_confidence(self):
        chunks = [_make_chunk("good", 0.9), _make_chunk("bad", 0.1)]
        retriever = _build_retriever(chunks)
        result = retriever.retrieve("query")
        assert result.retrieval_confidence == pytest.approx(0.9)


class TestWebFallback:
    def test_fires_only_when_all_chunks_incorrect(self):
        web_fallback = MagicMock()
        web_fallback.search.return_value = [_make_chunk("pubmed:1", CONFIDENCE_LOW)]

        chunks = [_make_chunk("bad", 0.1)]
        retriever = _build_retriever(chunks, web_fallback=web_fallback)
        result = retriever.retrieve("rare query with no local evidence")

        web_fallback.search.assert_called_once()
        assert result.used_web_fallback is True
        assert [c.chunk_id for c in result.chunks] == ["pubmed:1"]

    def test_does_not_fire_when_a_correct_chunk_exists(self):
        web_fallback = MagicMock()
        chunks = [_make_chunk("good", 0.9), _make_chunk("bad", 0.1)]
        retriever = _build_retriever(chunks, web_fallback=web_fallback)
        result = retriever.retrieve("query")

        web_fallback.search.assert_not_called()
        assert result.used_web_fallback is False

    def test_no_fallback_configured_returns_empty_chunks(self):
        chunks = [_make_chunk("bad", 0.1)]
        retriever = _build_retriever(chunks, web_fallback=None)
        result = retriever.retrieve("query")

        assert result.chunks == []
        assert result.retrieval_confidence == 0.0
        assert result.used_web_fallback is False

    def test_fallback_failure_leaves_empty_chunks_not_an_exception(self):
        web_fallback = MagicMock()
        web_fallback.search.return_value = []

        chunks = [_make_chunk("bad", 0.1)]
        retriever = _build_retriever(chunks, web_fallback=web_fallback)
        result = retriever.retrieve("query")

        assert result.chunks == []
        assert result.used_web_fallback is False
