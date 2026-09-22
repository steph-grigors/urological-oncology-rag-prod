"""
Main retrieval interface — the only class the generation layer imports.

Full pipeline per query:
  1. Embed query (OpenAI text-embedding-3-small)
  2. Dense search via Qdrant (top_k_retrieval, with metadata filters)
  3. BM25 search in-memory (top_k_retrieval, with metadata filters)
  4. RRF fusion → top_k_retrieval merged candidates
  5. Rerank via Cohere (or passthrough) → top_k_rerank final chunks
  6. Compute retrieval_confidence across the FULL reranked set (not only the
     chunks that survived grading, which put a floor under the number)

Metadata filters are applied before reranking:
  - Dense search: filters pushed down to Qdrant (most efficient)
  - BM25 search: filters applied on the in-memory corpus

RAGRetriever is constructed once at application startup and injected as a
FastAPI dependency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import logging

from openai import OpenAI

from config.constants import CONFIDENCE_LOW
from src.generation.confidence import compute_confidence
from src.db.vector_store import QdrantStore, ScoredChunk
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid import rrf_fusion
from src.retrieval.reranker import CohereReranker, RankedChunk
from src.retrieval.web_fallback import PubMedWebSearch

logger = logging.getLogger(__name__)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    query: str
    chunks: list[RankedChunk]
    retrieval_confidence: float   # mean relevance_score of returned chunks
    num_candidates: int           # candidates entering the reranker
    latency_ms: dict[str, float] = field(default_factory=dict)
    used_web_fallback: bool = False  # True if local evidence graded Incorrect and PubMed fallback fired


# ── Embedding cache ───────────────────────────────────────────────────────────
# Repeated identical queries skip the OpenAI call. Bounded and per-instance:
# see RAGRetriever._embed.

_EMBED_CACHE_MAXSIZE = 1000


# ── Retry helper ──────────────────────────────────────────────────────────────
# Wraps the two external network calls in retrieve() (OpenAI embeddings,
# Qdrant search) with a short exponential backoff. Catches broadly rather
# than enumerating each SDK's specific timeout/connection-error classes --
# a non-retryable error (bad API key, malformed request) just fails the
# same way after a couple of wasted short sleeps, which is an acceptable
# tradeoff for not having to track two SDKs' exception hierarchies here.
# Cohere reranking already has its own fallback (passthrough to RRF order
# on any failure, see reranker.py) and never raises, so it doesn't need
# this wrapper.

def _retry_with_backoff(fn, *args, attempts: int = 3, base_delay: float = 0.5, **kwargs):
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                    getattr(fn, "__qualname__", repr(fn)), attempt + 1, attempts, exc, delay,
                )
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc


# ── Diversity cap helper ──────────────────────────────────────────────────────

def _apply_diversity_cap(
    chunks: list[ScoredChunk],
    cap: int,
    capped_designs: frozenset[str],
) -> list[ScoredChunk]:
    """
    Return `chunks` with at most `cap` entries whose study_design is in
    `capped_designs`, preserving RRF rank order.

    Chunks beyond the cap are dropped entirely rather than demoted, so the
    reranker receives a smaller but higher-precision candidate set.
    """
    result: list[ScoredChunk] = []
    capped_count = 0
    for chunk in chunks:
        design = chunk.metadata.get("study_design", "unknown")
        if design in capped_designs:
            if capped_count >= cap:
                continue
            capped_count += 1
        result.append(chunk)
    return result


# ── RAGRetriever ──────────────────────────────────────────────────────────────

class RAGRetriever:
    """
    Combines QdrantStore + BM25Search + CohereReranker into a single
    `.retrieve()` call.

    Parameters
    ----------
    store                    : QdrantStore connected to the production collection
    bm25                     : pre-built BM25Search index
    reranker                 : CohereReranker (pass api_key="" for passthrough mode)
    openai_client            : OpenAI client for query embedding
    embedding_model          : must match the model used at index time
    top_k_retrieval          : candidates to fetch from each retriever
    top_k_rerank             : final chunks returned after reranking
    source_type_diversity_cap: max chunks with study_design = "review" that enter
                               the reranker. Forces primary trial evidence into the
                               top-N pool. Set to None to disable. Default 3 is
                               intentionally permissive for rare-cancer queries
                               (penile, adrenal) where reviews are the evidence base.
    web_fallback             : optional PubMedWebSearch. If every reranked chunk grades
                               Incorrect (relevance_score < CONFIDENCE_LOW), a single
                               question-level PubMed search runs instead of returning
                               nothing. Pass None (default) to disable.
    embed_cache_maxsize      : entries kept in the per-instance query-embedding cache.
                               0 disables caching entirely.
    """

    # Only narrative reviews are capped. "unknown" is excluded: it is the
    # fallback for papers whose study design couldn't be parsed from the
    # abstract — dropping them risks discarding valid RCTs or cohort studies
    # that simply had poorly-written abstracts.
    _CAPPED_DESIGNS: frozenset[str] = frozenset({"review"})

    def __init__(
        self,
        store: QdrantStore,
        bm25: BM25Search,
        reranker: CohereReranker,
        openai_client: OpenAI,
        embedding_model: str = "text-embedding-3-small",
        top_k_retrieval: int = 20,
        top_k_rerank: int = 5,
        source_type_diversity_cap: Optional[int] = 3,
        web_fallback: Optional[PubMedWebSearch] = None,
        embed_cache_maxsize: int = _EMBED_CACHE_MAXSIZE,
    ) -> None:
        self._store = store
        self._bm25 = bm25
        self._reranker = reranker
        self._openai = openai_client
        self._embedding_model = embedding_model
        self._top_k_retrieval = top_k_retrieval
        self._top_k_rerank = top_k_rerank
        self._diversity_cap = source_type_diversity_cap
        self._web_fallback = web_fallback
        self._embed_cache: dict[tuple[str, str], list[float]] = {}
        self._embed_cache_maxsize = embed_cache_maxsize

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
        dense_results = _retry_with_backoff(
            self._store.search_dense, query_embedding, k_ret, filters
        )
        timings["dense_ms"] = (time.perf_counter() - t1) * 1000

        # ── Step 3: BM25 search (filters applied in-memory) ──────────────
        t2 = time.perf_counter()
        bm25_results = self._bm25.search(query, k_ret, filters)
        timings["bm25_ms"] = (time.perf_counter() - t2) * 1000

        # ── Step 4: RRF fusion ────────────────────────────────────────────
        fused = rrf_fusion(dense_results, bm25_results, top_k=k_ret)

        # ── Step 4b: source diversity cap ────────────────────────────────
        # Prevent review/guideline chunks from occupying all reranker slots,
        # ensuring primary RCT evidence reaches the generation context.
        if self._diversity_cap is not None:
            fused = _apply_diversity_cap(fused, self._diversity_cap, self._CAPPED_DESIGNS)

        # ── Step 5: rerank ────────────────────────────────────────────────
        t3 = time.perf_counter()
        ranked = self._reranker.rerank(query, fused, top_n=k_rnk)
        timings["rerank_ms"] = (time.perf_counter() - t3) * 1000

        # ── Step 5b: cRAG-lite — discard chunks graded Incorrect ──────────
        # A chunk graded Incorrect (relevance_score < CONFIDENCE_LOW) is
        # dropped entirely rather than diluting the mean, so a single bad
        # match no longer silently lowers confidence for otherwise-good
        # retrieval. If every chunk was Incorrect, local evidence is too
        # weak to answer from, and a single question-level PubMed search
        # runs in its place (no per-chunk search, no agentic loop).
        graded = [c for c in ranked if c.relevance_score >= CONFIDENCE_LOW]
        # The set confidence is scored over: every candidate the reranker
        # judged, not only those that cleared the bar.
        candidates = ranked
        used_web_fallback = False
        if not graded and self._web_fallback is not None:
            t4 = time.perf_counter()
            # PubMed returns unranked search hits. Score them with the same
            # cross-encoder the local corpus goes through, rather than
            # assigning a constant: every abstract used to arrive at exactly
            # CONFIDENCE_LOW, so retrieval_confidence became exactly 0.45
            # whenever this path fired -- a number nobody had measured, on
            # results that had passed neither the quality gate nor a relevance
            # judgement.
            web_hits = self._web_fallback.search(query, max_results=k_rnk)
            if web_hits:
                web_ranked = self._reranker.rerank(query, web_hits, top_n=k_rnk)
                # Held to the same bar as local evidence. If the reranker says
                # these do not answer the question either, we are better off on
                # the ungrounded path, which at least says so.
                graded = [c for c in web_ranked if c.relevance_score >= CONFIDENCE_LOW]
                if graded:
                    # Confidence must describe the evidence actually returned.
                    # Leaving this as the local set would score the chunks we
                    # just discarded.
                    candidates = web_ranked
            timings["web_fallback_ms"] = (time.perf_counter() - t4) * 1000
            used_web_fallback = bool(graded)

        timings["total_ms"] = (time.perf_counter() - t_start) * 1000

        # Scored from the FULL reranked set, not from the survivors of the
        # cRAG filter above. Averaging only what survived a >= CONFIDENCE_LOW
        # cut meant the result could not fall below CONFIDENCE_LOW whenever
        # anything survived: a retrieval where 4 of 5 chunks were irrelevant
        # reported the score of the one that was not. Measured on production,
        # a penile sarcomatoid query reported 0.482 that way against an honest
        # 0.310 -- the difference between "hedged" and "caveated".
        #
        # compute_confidence rather than a bare mean, so the evidence-level,
        # single-paper and spread adjustments apply here too, and this value
        # and the generator's agree.
        confidence = compute_confidence(candidates).score if candidates else 0.0

        logger.info(
            "Retrieval timings — embed: %.0fms | dense: %.0fms | bm25: %.0fms | "
            "rerank: %.0fms | total: %.0fms | query_len: %d chars | web_fallback: %s",
            timings.get("embed_ms", 0),
            timings.get("dense_ms", 0),
            timings.get("bm25_ms", 0),
            timings.get("rerank_ms", 0),
            timings.get("total_ms", 0),
            len(query),
            used_web_fallback,
        )

        return RetrievalResult(
            query=query,
            chunks=graded,
            retrieval_confidence=confidence,
            num_candidates=len(fused),
            latency_ms=timings,
            used_web_fallback=used_web_fallback,
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _embed(self, query: str) -> list[float]:
        """Embed a query string, using a bounded in-process cache.

        The cache was previously an unbounded dict declared as a CLASS
        attribute, so it was shared by every RAGRetriever ever constructed and
        never evicted anything. In a long-running API each distinct query added
        a 1536-float list that was never reclaimed, which is a slow leak
        proportional to unique traffic rather than to anything bounded.

        Eviction is first-in-first-out rather than true least-recently-used: a
        hit does not move the entry, which keeps the container swappable for a
        plain dict and costs nothing that matters here, since the point is
        simply to bound the memory.
        """
        cache_key = (query, self._embedding_model)
        cached = self._embed_cache.get(cache_key)
        if cached is not None:
            return cached

        response = _retry_with_backoff(
            self._openai.embeddings.create,
            input=query,
            model=self._embedding_model,
        )
        embedding = response.data[0].embedding

        if self._embed_cache_maxsize > 0:
            while len(self._embed_cache) >= self._embed_cache_maxsize:
                self._embed_cache.pop(next(iter(self._embed_cache)))
            self._embed_cache[cache_key] = embedding
        return embedding
