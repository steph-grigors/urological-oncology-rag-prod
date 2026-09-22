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

    # The retriever now calls rerank twice when the fallback fires: once for
    # the local fused candidates, then again for the unranked PubMed hits, so
    # those arrive with a measured relevance instead of an assumed one. A fixed
    # return_value would hand the local chunks back on the second call, so the
    # double reranks whatever it is actually given from the second call on,
    # preserving each chunk's own score.
    def _rerank(query, chunks, top_n):
        if not _rerank.called:
            _rerank.called = True
            return reranked
        return [
            RankedChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                score=getattr(c, "relevance_score", c.score),
                relevance_score=getattr(c, "relevance_score", c.score),
                metadata=c.metadata,
            )
            for c in chunks[:top_n]
        ]

    _rerank.called = False
    reranker.rerank.side_effect = _rerank
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
        # Unranked, as PubMedWebSearch now returns them. The retriever is
        # responsible for scoring these, not the fallback.
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


class TestWebFallbackResultsAreScored:
    """PubMed hits used to arrive pre-assigned exactly CONFIDENCE_LOW, so
    retrieval_confidence became exactly 0.45 whenever this path fired -- a
    number nobody had measured, on results that passed neither the ingestion
    quality gate nor any relevance judgement. They now go through the same
    cross-encoder the local corpus does."""

    def test_hits_are_passed_through_the_reranker(self):
        web_fallback = MagicMock()
        web_fallback.search.return_value = [_make_chunk("pubmed:1", 0.8)]
        retriever = _build_retriever([_make_chunk("bad", 0.1)], web_fallback=web_fallback)

        retriever.retrieve("query with no local evidence")

        # Once for the local candidates, once for the PubMed hits.
        assert retriever._reranker.rerank.call_count == 2

    def test_confidence_reflects_the_measured_score(self):
        web_fallback = MagicMock()
        web_fallback.search.return_value = [_make_chunk("pubmed:1", 0.82)]
        retriever = _build_retriever([_make_chunk("bad", 0.1)], web_fallback=web_fallback)

        result = retriever.retrieve("query")

        assert result.used_web_fallback is True
        assert result.retrieval_confidence == pytest.approx(0.82)
        assert result.retrieval_confidence != pytest.approx(CONFIDENCE_LOW)

    def test_hits_the_reranker_rejects_are_dropped(self):
        """Held to the same bar as local evidence. If PubMed does not answer
        the question either, the ungrounded path at least says so."""
        web_fallback = MagicMock()
        web_fallback.search.return_value = [_make_chunk("pubmed:1", 0.2)]
        retriever = _build_retriever([_make_chunk("bad", 0.1)], web_fallback=web_fallback)

        result = retriever.retrieve("query")

        assert result.chunks == []
        assert result.used_web_fallback is False
        assert result.retrieval_confidence == 0.0

    def test_no_hits_leaves_the_ungrounded_path(self):
        web_fallback = MagicMock()
        web_fallback.search.return_value = []
        retriever = _build_retriever([_make_chunk("bad", 0.1)], web_fallback=web_fallback)

        result = retriever.retrieve("query")

        assert result.used_web_fallback is False
        assert result.retrieval_confidence == 0.0
