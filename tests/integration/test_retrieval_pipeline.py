"""
Integration tests for the retrieval pipeline.

Uses in-memory Qdrant and deterministic fake embeddings (no live APIs).
Fixtures are 10 JATS XML papers in tests/fixtures/sample_papers/.

Run with: pytest tests/integration/ -v
Excluded from default pytest run via the `integration` marker in pytest.ini.
"""

from __future__ import annotations

import pytest

from src.db.vector_store import QdrantStore, ScoredChunk
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid import rrf_fusion
from src.retrieval.reranker import CohereReranker
from tests.integration.conftest import query_embedding_for


pytestmark = pytest.mark.integration


# ── QdrantStore / dense search ────────────────────────────────────────────────

class TestDenseSearch:
    """Dense vector search against the in-memory Qdrant collection."""

    def test_returns_non_empty_results(
        self, qdrant_store: QdrantStore, indexed_chunks
    ):
        embedding = query_embedding_for("prostate")
        results = qdrant_store.search_dense(embedding, top_k=5)
        assert len(results) > 0

    def test_result_fields_populated(
        self, qdrant_store: QdrantStore, indexed_chunks
    ):
        embedding = query_embedding_for("prostate")
        results = qdrant_store.search_dense(embedding, top_k=3)
        for r in results:
            assert isinstance(r.chunk_id, str)
            assert len(r.text) > 0
            assert isinstance(r.score, float)
            assert "cancer_type" in r.metadata

    def test_cancer_type_filter_excludes_other_types(
        self, qdrant_store: QdrantStore, indexed_chunks
    ):
        """Filtering by cancer_type=prostate must return no kidney chunks."""
        embedding = query_embedding_for("prostate")
        results = qdrant_store.search_dense(
            embedding, top_k=20, filters={"cancer_type": ["prostate"]}
        )
        for r in results:
            assert "prostate" in r.metadata.get("cancer_type", [])

    def test_year_min_filter_excludes_old_papers(
        self, qdrant_store: QdrantStore, indexed_chunks
    ):
        embedding = query_embedding_for("bladder")
        results = qdrant_store.search_dense(
            embedding, top_k=20, filters={"year_min": 2021}
        )
        for r in results:
            year = r.metadata.get("year")
            if year is not None:
                assert year >= 2021, f"Got year {year} after year_min=2021 filter"

    def test_study_design_filter(
        self, qdrant_store: QdrantStore, indexed_chunks
    ):
        embedding = query_embedding_for("prostate")
        results = qdrant_store.search_dense(
            embedding, top_k=20, filters={"study_design": ["rct"]}
        )
        for r in results:
            assert r.metadata.get("study_design") == "rct"

    def test_same_cancer_type_chunks_score_higher(
        self, qdrant_store: QdrantStore, indexed_chunks
    ):
        """Prostate query embedding should rank prostate chunks above bladder."""
        prostate_emb = query_embedding_for("prostate")
        results = qdrant_store.search_dense(prostate_emb, top_k=20)
        top5_cancer_types = [
            ct
            for r in results[:5]
            for ct in r.metadata.get("cancer_type", [])
        ]
        prostate_count = top5_cancer_types.count("prostate")
        assert prostate_count >= 3, (
            f"Expected prostate chunks to dominate top-5, got {prostate_count}/5"
        )

    def test_collection_stats_reflects_indexed_chunks(
        self, qdrant_store: QdrantStore, indexed_chunks
    ):
        stats = qdrant_store.collection_stats()
        assert stats["point_count"] == len(indexed_chunks)
        assert stats["dense_vector_size"] == 1536


# ── BM25 search ───────────────────────────────────────────────────────────────

class TestBM25Search:
    """BM25 exact-term matching over the fixture corpus."""

    def test_enzalutamide_query_top3_contains_term(
        self, bm25_index: BM25Search
    ):
        """Chunks mentioning 'enzalutamide' must appear in top 3 for that query."""
        results = bm25_index.search("enzalutamide", top_k=3)
        assert len(results) >= 1
        top_texts = [r.text.lower() for r in results[:3]]
        assert any("enzalutamide" in t for t in top_texts), (
            "None of the top-3 BM25 results mention 'enzalutamide'"
        )

    def test_sunitinib_query_returns_kidney_chunks(
        self, bm25_index: BM25Search
    ):
        results = bm25_index.search("sunitinib renal cell carcinoma", top_k=5)
        top_texts = [r.text.lower() for r in results[:3]]
        assert any("sunitinib" in t for t in top_texts)

    def test_bep_chemotherapy_query_returns_testicular_chunks(
        self, bm25_index: BM25Search
    ):
        results = bm25_index.search("BEP bleomycin etoposide cisplatin testicular", top_k=5)
        top_texts = [r.text.lower() for r in results[:3]]
        assert any("bep" in t or "bleomycin" in t for t in top_texts)

    def test_filter_by_rct_excludes_case_reports(
        self, bm25_index: BM25Search
    ):
        """Filtering by study_design=rct must exclude case_report chunks."""
        results = bm25_index.search(
            "prostate cancer treatment",
            top_k=20,
            filters={"study_design": ["rct"]},
        )
        for r in results:
            assert r.metadata.get("study_design") != "case_report", (
                f"case_report chunk {r.chunk_id} leaked through rct filter"
            )

    def test_filter_by_cancer_type_prostate_excludes_bladder(
        self, bm25_index: BM25Search
    ):
        results = bm25_index.search(
            "cancer chemotherapy",
            top_k=20,
            filters={"cancer_type": ["prostate"]},
        )
        for r in results:
            cancer = r.metadata.get("cancer_type", [])
            assert "prostate" in cancer


# ── RRF fusion ────────────────────────────────────────────────────────────────

class TestRRFFusion:
    """RRF fusion over the fixture corpus vs. dense-only baseline."""

    def test_rrf_returns_more_relevant_results_than_dense_only(
        self, qdrant_store: QdrantStore, bm25_index: BM25Search
    ):
        """
        For an enzalutamide query, RRF should return results that include
        the BM25-identified enzalutamide chunks even if they score low on
        the embedding similarity alone.

        We construct a query embedding pointing toward 'kidney' (off-topic
        for enzalutamide) so dense-only misses the enzalutamide chunks,
        while BM25 still finds them — and RRF recovers them.
        """
        # Dense: use kidney embedding → misses prostate/enzalutamide chunks
        kidney_emb = query_embedding_for("kidney")
        dense_results = qdrant_store.search_dense(kidney_emb, top_k=20)

        # BM25: exact term match → finds enzalutamide chunks
        bm25_results = bm25_index.search("enzalutamide", top_k=20)

        fused = rrf_fusion(dense_results, bm25_results, top_k=20)

        # Count enzalutamide mentions in top-5 results of each approach
        def enzalutamide_count(chunks: list[ScoredChunk], n: int) -> int:
            return sum(
                1 for c in chunks[:n] if "enzalutamide" in c.text.lower()
            )

        dense_hits = enzalutamide_count(dense_results, 5)
        fused_hits = enzalutamide_count(fused, 5)

        assert fused_hits > dense_hits, (
            f"RRF ({fused_hits}) should exceed dense-only ({dense_hits}) "
            "for enzalutamide when dense query is off-topic"
        )

    def test_rrf_output_deduplicated(
        self, qdrant_store: QdrantStore, bm25_index: BM25Search
    ):
        emb = query_embedding_for("prostate")
        dense = qdrant_store.search_dense(emb, top_k=10)
        bm25 = bm25_index.search("prostate cancer", top_k=10)
        fused = rrf_fusion(dense, bm25)
        ids = [c.chunk_id for c in fused]
        assert len(ids) == len(set(ids))

    def test_rrf_sorted_descending(
        self, qdrant_store: QdrantStore, bm25_index: BM25Search
    ):
        emb = query_embedding_for("bladder")
        dense = qdrant_store.search_dense(emb, top_k=10)
        bm25 = bm25_index.search("bladder cancer cisplatin", top_k=10)
        fused = rrf_fusion(dense, bm25)
        scores = [c.score for c in fused]
        assert scores == sorted(scores, reverse=True)


# ── Reranker (no Cohere key → passthrough) ───────────────────────────────────

class TestRerankerDegradation:
    """Reranker falls back gracefully when no API key is configured."""

    def test_passthrough_when_no_api_key(
        self, bm25_index: BM25Search
    ):
        reranker = CohereReranker(api_key="")
        assert not reranker.is_available()

        chunks = bm25_index.search("enzalutamide", top_k=5)
        assert len(chunks) > 0

        scored = [
            c.__class__(chunk_id=c.chunk_id, text=c.text,
                        score=c.score, metadata=c.metadata)
            for c in chunks
        ]
        ranked = reranker.rerank("enzalutamide treatment", scored, top_n=3)
        assert len(ranked) == min(3, len(chunks))
        for r in ranked:
            assert r.relevance_score == r.score  # passthrough: scores equal

    def test_passthrough_preserves_rrf_ordering(
        self, qdrant_store: QdrantStore, bm25_index: BM25Search
    ):
        reranker = CohereReranker(api_key="")
        emb = query_embedding_for("prostate")
        dense = qdrant_store.search_dense(emb, top_k=10)
        bm25 = bm25_index.search("prostate cancer enzalutamide", top_k=10)
        fused = rrf_fusion(dense, bm25, top_k=10)
        ranked = reranker.rerank("prostate cancer enzalutamide", fused, top_n=5)

        # Without Cohere, order must match the input fused order (first top_n)
        fused_ids_top5 = [c.chunk_id for c in fused[:5]]
        ranked_ids = [r.chunk_id for r in ranked]
        assert ranked_ids == fused_ids_top5


# ── delete_by_pmid ────────────────────────────────────────────────────────────

class TestDeleteByPmid:
    """Deletion removes all chunks for a paper and nothing else."""

    def test_delete_removes_target_paper(
        self, qdrant_store: QdrantStore, indexed_chunks
    ):
        # PMC1004 is the case-report paper — smallest footprint
        before = qdrant_store.collection_stats()["point_count"]
        qdrant_store.delete_by_pmid("31001004")
        after = qdrant_store.collection_stats()["point_count"]
        assert after < before, "Point count should decrease after deletion"

    def test_deleted_paper_absent_from_dense_search(
        self, qdrant_store: QdrantStore, indexed_chunks
    ):
        emb = query_embedding_for("prostate")
        results = qdrant_store.search_dense(
            emb, top_k=50, filters={"study_design": ["case_report"]}
        )
        pmcids = [r.metadata.get("pmcid") for r in results]
        assert "1004" not in pmcids
