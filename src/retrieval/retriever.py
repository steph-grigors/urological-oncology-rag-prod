"""
Main retrieval interface — the only class the generation layer should import.

`Retriever` composes VectorSearch + BM25Search + HybridSearch + Reranker
into a single `.retrieve()` method.  It is constructed once at application
startup and injected as a FastAPI dependency.

Full pipeline per query:
    1. HybridSearch.search(query, top_k=TOP_K_RETRIEVAL, filter=filter)
       → fused list of up to 20 candidates
    2. Reranker.rerank(query, candidates, top_k=TOP_K_RERANK)
       → top 5 chunks by cross-encoder + evidence-grade score
    3. Confidence signal: compute mean rerank score across top-k chunks
       → passed to generator as `retrieval_confidence`

The retriever does NOT generate answers — that is `generation.generator`.
It returns a `RetrievalResult` that the generator consumes.

Public API (to be implemented):
    class Retriever:
        def __init__(
            self,
            vector_search: VectorSearch,
            bm25_search: BM25Search,
            reranker: Reranker,
            settings: Settings,
        ): ...

        def retrieve(
            self,
            query: str,
            filter: SearchFilter | None = None,
            top_k_retrieval: int | None = None,
            top_k_rerank: int | None = None,
        ) -> RetrievalResult: ...

    RetrievalResult(dataclass)
        query: str
        chunks: list[SearchResult]     # final reranked chunks
        retrieval_confidence: float    # mean rerank score of top-k
        num_candidates: int            # total candidates before rerank
        latency_ms: dict[str, float]   # per-step timing
"""
