"""
Reciprocal Rank Fusion (RRF) of dense and sparse retrieval results.

RRF is a parameter-free rank fusion method that is robust to score-scale
differences between the two retrievers:

    RRF_score(d) = Σ  1 / (k + rank_i(d))

where k = RRF_K (default 60, from constants.py) and rank_i is the rank
assigned by retriever i.  Documents not present in a retriever's results
are treated as having rank = ∞.

Design decisions:
    - No learned weights — RRF is used intentionally to avoid overfitting
      fusion parameters on a small evaluation set.
    - `top_k` for each underlying retriever is set to `TOP_K_RETRIEVAL`
      (default 20); after fusion the top `TOP_K_RETRIEVAL` fused results
      are passed to the reranker.
    - Deduplication by chunk_id before fusion: if the same chunk appears
      in both lists the higher rank is used per-retriever.

Public API (to be implemented):
    def reciprocal_rank_fusion(
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        k: int = RRF_K,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        Return a merged, re-ranked list of SearchResult objects with a
        new `score` field representing the RRF score.

    class HybridSearch:
        def __init__(
            self,
            vector_search: VectorSearch,
            bm25_search: BM25Search,
            settings: Settings,
        ): ...

        def search(
            self,
            query: str,
            top_k: int | None = None,
            filter: SearchFilter | None = None,
        ) -> list[SearchResult]: ...
"""
