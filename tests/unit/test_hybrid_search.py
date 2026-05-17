"""
Unit tests for src/retrieval/hybrid.py (rrf_fusion) and
src/retrieval/bm25_search.py (BM25Search).

All tests are pure in-memory — no Qdrant, no network, no OpenAI.

Run: pytest tests/unit/test_hybrid_search.py -v
"""

from __future__ import annotations

import pytest

from src.db.vector_store import ScoredChunk
from src.retrieval.bm25_search import BM25Search, _tokenize
from src.retrieval.hybrid import rrf_fusion


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk(
    chunk_id: str,
    text: str = "",
    score: float = 0.0,
    metadata: dict | None = None,
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        metadata=metadata or {},
    )


def _ranked_ids(chunks: list[ScoredChunk]) -> list[str]:
    return [c.chunk_id for c in chunks]


# ── RRF scoring ───────────────────────────────────────────────────────────────

class TestRRFScoring:
    """Score arithmetic and rank-assignment rules."""

    def test_document_in_both_lists_higher_score_than_single_list(self):
        """A chunk appearing in both lists must outscore one present in only one."""
        dense = [_chunk("A"), _chunk("B"), _chunk("C")]
        bm25  = [_chunk("A"), _chunk("D"), _chunk("E")]
        result = rrf_fusion(dense, bm25)
        scores = {c.chunk_id: c.score for c in result}
        # A is rank-1 in both; B is rank-2 dense only; D is rank-2 bm25 only
        assert scores["A"] > scores["B"]
        assert scores["A"] > scores["D"]

    def test_document_only_in_dense_uses_fallback_bm25_rank(self):
        dense = [_chunk("X"), _chunk("Y")]
        bm25  = [_chunk("Z")]
        result = rrf_fusion(dense, bm25)
        scores = {c.chunk_id: c.score for c in result}
        # X is rank-1 dense; Z is rank-1 bm25 — X gets fallback bm25 rank=2
        x_score = 1 / (60 + 1) + 1 / (60 + 3)   # rank1 dense, fallback rank=3
        z_score = 1 / (60 + 3) + 1 / (60 + 1)   # fallback rank=3 dense, rank1 bm25
        # X and Z should have identical scores (symmetric fallback)
        assert abs(scores["X"] - scores["Z"]) < 1e-12

    def test_document_only_in_sparse_uses_fallback_dense_rank(self):
        dense = [_chunk("A")]
        bm25  = [_chunk("A"), _chunk("B")]
        result = rrf_fusion(dense, bm25)
        scores = {c.chunk_id: c.score for c in result}
        # B is rank-2 bm25 only; fallback dense rank = 2
        # A is rank-1 in both
        assert scores["A"] > scores["B"]

    def test_rrf_k_affects_score(self):
        """Smaller k produces higher RRF scores (scores are more spread)."""
        dense = [_chunk("A")]
        bm25  = [_chunk("A")]
        result_k10  = rrf_fusion(dense, bm25, k=10)
        result_k100 = rrf_fusion(dense, bm25, k=100)
        assert result_k10[0].score > result_k100[0].score

    def test_score_formula_is_correct(self):
        """Verify exact arithmetic: A is rank-1 in both with k=60."""
        dense = [_chunk("A"), _chunk("B")]
        bm25  = [_chunk("A"), _chunk("C")]
        result = rrf_fusion(dense, bm25, k=60)
        a = next(c for c in result if c.chunk_id == "A")
        expected = 1 / (60 + 1) + 1 / (60 + 1)
        assert abs(a.score - expected) < 1e-12

    def test_fallback_rank_equals_max_list_length_plus_one(self):
        """Fallback rank = max(len(dense), len(bm25)) + 1."""
        dense = [_chunk("A"), _chunk("B"), _chunk("C")]  # len=3
        bm25  = [_chunk("D")]                             # len=1; fallback=4
        result = rrf_fusion(dense, bm25, k=60)
        scores = {c.chunk_id: c.score for c in result}
        # D: rank-1 bm25, fallback rank-4 dense → 1/64 + 1/61
        expected_d = 1 / (60 + 4) + 1 / (60 + 1)
        assert abs(scores["D"] - expected_d) < 1e-12


# ── Fusion edge cases ─────────────────────────────────────────────────────────

class TestFusionEdgeCases:
    """Behaviour with empty inputs and duplicates."""

    def test_empty_dense_returns_sparse_chunks(self):
        bm25 = [_chunk("X", score=0.9), _chunk("Y", score=0.5)]
        result = rrf_fusion([], bm25)
        assert _ranked_ids(result) == ["X", "Y"]

    def test_empty_sparse_returns_dense_chunks(self):
        dense = [_chunk("A", score=0.8), _chunk("B", score=0.4)]
        result = rrf_fusion(dense, [])
        assert _ranked_ids(result) == ["A", "B"]

    def test_both_empty_returns_empty(self):
        assert rrf_fusion([], []) == []

    def test_deduplication_same_chunk_in_both_lists(self):
        """The same chunk_id must appear exactly once in the output."""
        dense = [_chunk("A"), _chunk("B")]
        bm25  = [_chunk("A"), _chunk("C")]
        result = rrf_fusion(dense, bm25)
        ids = _ranked_ids(result)
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids in output"
        assert "A" in ids

    def test_complete_overlap_deduplicates_all(self):
        dense = [_chunk("A"), _chunk("B"), _chunk("C")]
        bm25  = [_chunk("C"), _chunk("B"), _chunk("A")]
        result = rrf_fusion(dense, bm25)
        assert len(result) == 3


# ── Output format ─────────────────────────────────────────────────────────────

class TestOutputFormat:
    """Sorting and top_k truncation."""

    def test_sorted_descending_by_score(self):
        dense = [_chunk("A"), _chunk("B"), _chunk("C")]
        bm25  = [_chunk("C"), _chunk("B"), _chunk("A")]
        result = rrf_fusion(dense, bm25)
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_truncates_output(self):
        dense = [_chunk(str(i)) for i in range(10)]
        bm25  = [_chunk(str(i)) for i in range(10, 20)]
        result = rrf_fusion(dense, bm25, top_k=5)
        assert len(result) == 5

    def test_top_k_none_returns_all(self):
        dense = [_chunk(str(i)) for i in range(5)]
        bm25  = [_chunk(str(i + 5)) for i in range(5)]
        result = rrf_fusion(dense, bm25, top_k=None)
        assert len(result) == 10

    def test_metadata_preserved_from_source_list(self):
        meta = {"section": "results", "cancer_type": ["prostate"]}
        dense = [_chunk("A", metadata=meta)]
        result = rrf_fusion(dense, [])
        assert result[0].metadata == meta

    def test_text_preserved_from_source_list(self):
        dense = [_chunk("A", text="enzalutamide progression")]
        result = rrf_fusion(dense, [])
        assert result[0].text == "enzalutamide progression"


# ── BM25Search unit tests ─────────────────────────────────────────────────────

class TestBM25Search:
    """In-memory BM25 behaviour without Qdrant."""

    _CORPUS = [
        _chunk("enzalutamide_1",
               text="Enzalutamide significantly improved overall survival in mCRPC patients",
               metadata={"cancer_type": ["prostate"], "section": "results",
                          "study_design": "rct", "evidence_level": 2}),
        _chunk("enzalutamide_2",
               text="Enzalutamide versus placebo: progression-free survival benefit confirmed",
               metadata={"cancer_type": ["prostate"], "section": "results",
                          "study_design": "rct", "evidence_level": 2}),
        _chunk("sunitinib_1",
               text="Sunitinib remains the benchmark comparator in renal cell carcinoma trials",
               metadata={"cancer_type": ["kidney"], "section": "introduction",
                          "study_design": "review", "evidence_level": 5}),
        _chunk("bladder_1",
               text="Cisplatin gemcitabine chemotherapy for metastatic urothelial carcinoma bladder",
               metadata={"cancer_type": ["bladder"], "section": "methods",
                          "study_design": "cohort", "evidence_level": 3,
                          "year": 2019}),
        _chunk("bladder_rct",
               text="Pembrolizumab randomised controlled trial bladder cancer immunotherapy",
               metadata={"cancer_type": ["bladder"], "section": "results",
                          "study_design": "rct", "evidence_level": 2,
                          "year": 2022}),
    ]

    def _bm25(self) -> BM25Search:
        return BM25Search(list(self._CORPUS))

    def test_drug_name_query_returns_relevant_chunk_in_top_results(self):
        bm25 = self._bm25()
        results = bm25.search("enzalutamide", top_k=3)
        top_ids = _ranked_ids(results)
        assert any("enzalutamide" in cid for cid in top_ids[:2])

    def test_exact_term_in_top_3(self):
        bm25 = self._bm25()
        results = bm25.search("enzalutamide overall survival", top_k=3)
        texts = [c.text.lower() for c in results]
        assert any("enzalutamide" in t for t in texts)

    def test_unrelated_query_scores_zero_documents(self):
        bm25 = self._bm25()
        results = bm25.search("zzzzzzz nonexistent_term_xyzxyz", top_k=5)
        # All scores should be 0 so no results returned
        assert results == []

    def test_filter_by_cancer_type(self):
        bm25 = self._bm25()
        results = bm25.search("cancer treatment", top_k=10,
                               filters={"cancer_type": ["kidney"]})
        for c in results:
            ct = c.metadata.get("cancer_type", [])
            assert "kidney" in ct

    def test_filter_by_study_design_excludes_other_designs(self):
        bm25 = self._bm25()
        results = bm25.search("cancer treatment", top_k=10,
                               filters={"study_design": ["rct"]})
        for c in results:
            assert c.metadata.get("study_design") == "rct"

    def test_filter_by_year_min(self):
        bm25 = self._bm25()
        results = bm25.search("bladder cancer", top_k=10,
                               filters={"year_min": 2021})
        for c in results:
            year = c.metadata.get("year")
            if year is not None:
                assert year >= 2021

    def test_size_reflects_corpus(self):
        bm25 = self._bm25()
        assert bm25.size == len(self._CORPUS)

    def test_build_replaces_index(self):
        # Use 3-doc corpus so Robertson IDF is positive (log(2.5/1.5) > 0).
        # A 1-doc corpus yields IDF = log(0.5/1.5) < 0, clamped to 0 — no hits.
        bm25 = self._bm25()
        new_corpus = [
            _chunk("target", text="nivolumab checkpoint inhibitor immunotherapy"),
            _chunk("other_a", text="enzalutamide prostate androgen receptor"),
            _chunk("other_b", text="pembrolizumab bladder urothelial carcinoma"),
        ]
        bm25.build(new_corpus)
        assert bm25.size == 3
        results = bm25.search("nivolumab", top_k=5)
        assert results[0].chunk_id == "target"

    def test_empty_corpus_returns_empty(self):
        bm25 = BM25Search([])
        assert bm25.search("anything", top_k=5) == []

    def test_empty_query_returns_empty(self):
        bm25 = self._bm25()
        assert bm25.search("", top_k=5) == []

    def test_top_k_respected(self):
        bm25 = self._bm25()
        results = bm25.search("cancer", top_k=2)
        assert len(results) <= 2


# ── Tokeniser ─────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_lowercases(self):
        assert "enzalutamide" in _tokenize("ENZALUTAMIDE")

    def test_strips_punctuation(self):
        tokens = _tokenize("cancer, therapy; outcome.")
        assert "cancer" in tokens
        assert "therapy" in tokens
        assert "outcome" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_no_empty_tokens(self):
        for t in _tokenize("  lots   of   spaces  "):
            assert t.strip() != ""
