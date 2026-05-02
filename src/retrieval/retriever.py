"""
Main retrieval interface — the only class the generation layer imports.

Full pipeline per query:
  1. Embed query (OpenAI text-embedding-3-small)
  2. Dense search via Qdrant (top_k_retrieval, with metadata filters)
  3. BM25 search in-memory (top_k_retrieval, with metadata filters)
  4. RRF fusion → top_k_retrieval merged candidates
  5. Rerank via Cohere (or passthrough) → top_k_rerank final chunks
  6. Compute retrieval_confidence = mean relevance_score across final chunks

Metadata filters are applied before reranking:
  - Dense search: filters pushed down to Qdrant (most efficient)
  - BM25 search: filters applied on the in-memory corpus

RAGRetriever is constructed once at application startup and injected as a
FastAPI dependency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from openai import OpenAI

from src.db.vector_store import QdrantStore, ScoredChunk
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid import rrf_fusion
from src.retrieval.reranker import CohereReranker, RankedChunk


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    query: str
    chunks: list[RankedChunk]
    retrieval_confidence: float   # mean relevance_score of returned chunks
    num_candidates: int           # candidates entering the reranker
    latency_ms: dict[str, float] = field(default_factory=dict)


# ── Embedding cache ───────────────────────────────────────────────────────────
# Module-level LRU so repeated identical queries skip the API call.

@lru_cache(maxsize=1000)
def _cached_embed(query: str, model: str, client_id: int) -> tuple[float, ...]:
    raise NotImplementedError  # replaced at runtime by RAGRetriever._embed


# ── RAGRetriever ──────────────────────────────────────────────────────────────

class RAGRetriever:
    """
    Combines QdrantStore + BM25Search + CohereReranker into a single
    `.retrieve()` call.

    Parameters
    ----------
    store           : QdrantStore connected to the production collection
    bm25            : pre-built BM25Search index
    reranker        : CohereReranker (pass api_key="" for passthrough mode)
    openai_client   : OpenAI client for query embedding
    embedding_model : must match the model used at index time
    top_k_retrieval : candidates to fetch from each retriever
    top_k_rerank    : final chunks returned after reranking
    """

    def __init__(
        self,
        store: QdrantStore,
        bm25: BM25Search,
        reranker: CohereReranker,
        openai_client: OpenAI,
        embedding_model: str = "text-embedding-3-small",
        top_k_retrieval: int = 20,
        top_k_rerank: int = 5,
    ) -> None:
        self._store = store
        self._bm25 = bm25
        self._reranker = reranker
        self._openai = openai_client
        self._embedding_model = embedding_model
        self._top_k_retrieval = top_k_retrieval
        self._top_k_rerank = top_k_rerank

    # ── Public API ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        filters: Optional[dict] = None,
        top_k_retrieval: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Run the full retrieval pipeline for `query`.

        filters dict keys (all optional):
            cancer_type       list[str]
            section           list[str]
            study_design      list[str]
            chunk_type        list[str]
            year_min          int
            year_max          int
            evidence_level_max int
        """
        k_ret = top_k_retrieval or self._top_k_retrieval
        k_rnk = top_k_rerank or self._top_k_rerank
        timings: dict[str, float] = {}
        t_start = time.perf_counter()

        # ── Step 1: embed query ───────────────────────────────────────────
        query_embedding = self._embed(query)
        timings["embed_ms"] = (time.perf_counter() - t_start) * 1000

        # ── Step 2: dense search (filters applied in Qdrant) ─────────────
        t1 = time.perf_counter()
        dense_results = self._store.search_dense(query_embedding, k_ret, filters)
        timings["dense_ms"] = (time.perf_counter() - t1) * 1000

        # ── Step 3: BM25 search (filters applied in-memory) ──────────────
        t2 = time.perf_counter()
        bm25_results = self._bm25.search(query, k_ret, filters)
        timings["bm25_ms"] = (time.perf_counter() - t2) * 1000

        # ── Step 4: RRF fusion ────────────────────────────────────────────
        fused = rrf_fusion(dense_results, bm25_results, top_k=k_ret)

        # ── Step 5: rerank ────────────────────────────────────────────────
        t3 = time.perf_counter()
        ranked = self._reranker.rerank(query, fused, top_n=k_rnk)
        timings["rerank_ms"] = (time.perf_counter() - t3) * 1000
        timings["total_ms"] = (time.perf_counter() - t_start) * 1000

        confidence = (
            sum(c.relevance_score for c in ranked) / len(ranked)
            if ranked else 0.0
        )

        return RetrievalResult(
            query=query,
            chunks=ranked,
            retrieval_confidence=confidence,
            num_candidates=len(fused),
            latency_ms=timings,
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _embed(self, query: str) -> list[float]:
        """Embed a query string, using an in-process LRU cache."""
        cache_key = (query, self._embedding_model)
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]
        response = self._openai.embeddings.create(
            input=query,
            model=self._embedding_model,
        )
        embedding = response.data[0].embedding
        self._embed_cache[cache_key] = embedding
        return embedding

    # Simple dict-based cache (module lru_cache doesn't work on instance methods)
    _embed_cache: dict = {}
