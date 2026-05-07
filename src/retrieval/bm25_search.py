"""
In-memory BM25 keyword search using the rank_bm25 library.

BM25 complements dense vector search by excelling at exact clinical term
matching (drug names, gene symbols, numeric thresholds) that embeddings
sometimes miss or mis-rank.

Usage:
    # Build from a pre-loaded list of ScoredChunk objects
    bm25 = BM25Search(chunks)

    # Or load the entire Qdrant collection at startup (with disk cache)
    bm25 = BM25Search.from_qdrant(store)

    results = bm25.search("enzalutamide progression-free survival", top_k=20)

Disk cache:
    The tokenized corpus is persisted to BM25_CACHE_PATH (default
    data/bm25_cache.pkl) so restarts load in seconds instead of scrolling
    685K chunks from Qdrant.  Call save() after (re-)building to update it.
    The cache is invalidated automatically when the Qdrant point count changes.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi

from src.db.vector_store import ScoredChunk

if TYPE_CHECKING:
    from src.db.vector_store import QdrantStore

logger = logging.getLogger(__name__)

BM25_CACHE_PATH = "data/bm25_cache.pkl"


# ── Tokeniser (shared with sparse vector helpers) ─────────────────────────────

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if w]


# ── BM25Search ────────────────────────────────────────────────────────────────

class BM25Search:
    """
    In-memory BM25 index over all indexed chunks.

    The index is built once at startup and held in memory.  The tokenized
    corpus is persisted to disk (BM25_CACHE_PATH) so restarts load in seconds
    instead of re-scrolling Qdrant.  The cache is invalidated when the Qdrant
    point count no longer matches the cached count.
    """

    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self._chunks: list[ScoredChunk] = []
        self._bm25: BM25Okapi | None = None
        if chunks:
            self.build(chunks)

    @classmethod
    def from_qdrant(
        cls,
        store: "QdrantStore",
        cache_path: str = BM25_CACHE_PATH,
    ) -> "BM25Search":
        """Load BM25 from disk cache if valid, otherwise scroll Qdrant and save."""
        live_count = store.count()
        cached = cls._load_cache(cache_path, live_count)
        if cached is not None:
            logger.info("BM25 loaded from cache (%d chunks)", cached.size)
            return cached

        logger.info("BM25 cache miss (live=%d) — scrolling Qdrant…", live_count)
        chunks = store.scroll_all()
        instance = cls(chunks)
        instance.save(cache_path, live_count)
        return instance

    # ── Disk cache ────────────────────────────────────────────────────────

    def save(self, cache_path: str = BM25_CACHE_PATH, count: int | None = None) -> None:
        """Persist the index to disk.  count is stored for invalidation."""
        try:
            path = Path(cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "count": count if count is not None else len(self._chunks),
                "chunks": self._chunks,
                "bm25": self._bm25,
            }
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.rename(path)
            logger.info("BM25 cache saved to %s (%d chunks)", cache_path, len(self._chunks))
        except Exception as exc:
            logger.warning("Could not save BM25 cache: %s", exc)

    @classmethod
    def _load_cache(cls, cache_path: str, live_count: int) -> "BM25Search | None":
        path = Path(cache_path)
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
            if payload.get("count") != live_count:
                logger.info(
                    "BM25 cache stale (cached=%d, live=%d)",
                    payload.get("count"), live_count,
                )
                return None
            instance = cls.__new__(cls)
            instance._chunks = payload["chunks"]
            instance._bm25 = payload["bm25"]
            return instance
        except Exception as exc:
            logger.warning("Could not load BM25 cache: %s", exc)
            return None

    # ── Index management ──────────────────────────────────────────────────

    def build(self, chunks: list[ScoredChunk]) -> None:
        """(Re-)build the BM25 index from a chunk list."""
        self._chunks = list(chunks)
        corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)

    @property
    def size(self) -> int:
        return len(self._chunks)

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[ScoredChunk]:
        """
        Return the top_k chunks by BM25 score.

        filters dict supports the same keys as QdrantStore._build_filter:
        cancer_type, section, study_design, chunk_type, year_min, year_max,
        evidence_level_max.  Filtering is applied post-scoring.
        """
        if self._bm25 is None or not self._chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # For long queries, keep only the top-N tokens by IDF score.
        # High-IDF tokens are rare in the corpus = most clinically specific.
        # This caps BM25 scoring cost at O(N × corpus_size) regardless of query length.
        _MAX_TOKENS = 20
        unique_tokens = list(dict.fromkeys(query_tokens))
        if len(unique_tokens) > _MAX_TOKENS:
            unique_tokens = sorted(
                unique_tokens,
                key=lambda t: self._bm25.idf.get(t, 0.0),
                reverse=True,
            )[:_MAX_TOKENS]
            query_tokens = unique_tokens

        scores: np.ndarray = self._bm25.get_scores(query_tokens)

        # score == 0.0 means the document contains none of the query tokens.
        # score < 0 can occur in small corpora where IDF is negative (term
        # appears in all documents); those chunks are still relevant matches.
        nonzero_indices = np.nonzero(scores)[0]
        if len(nonzero_indices) == 0:
            return []

        # Sort the matching subset by score descending (highest = most relevant)
        order = np.argsort(scores[nonzero_indices])[::-1]
        ranked_indices = nonzero_indices[order]

        results: list[ScoredChunk] = []
        for idx in ranked_indices:
            chunk = self._chunks[idx]
            if filters and not _matches_filters(chunk, filters):
                continue
            results.append(ScoredChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(scores[idx]),
                metadata=chunk.metadata,
            ))
            if len(results) >= top_k:
                break

        return results


# ── Filter helper ─────────────────────────────────────────────────────────────

def _matches_filters(chunk: ScoredChunk, filters: dict) -> bool:
    """Return True if the chunk metadata satisfies all filter conditions."""
    meta = chunk.metadata

    for key in ("section", "study_design", "chunk_type"):
        if key in filters:
            allowed = filters[key]
            if isinstance(allowed, str):
                allowed = [allowed]
            if meta.get(key) not in allowed:
                return False

    if "cancer_type" in filters:
        allowed_ct = filters["cancer_type"]
        if isinstance(allowed_ct, str):
            allowed_ct = [allowed_ct]
        chunk_ct = meta.get("cancer_type") or []
        if isinstance(chunk_ct, str):
            chunk_ct = [chunk_ct]
        if not any(ct in allowed_ct for ct in chunk_ct):
            return False

    year = meta.get("year")
    if "year_min" in filters and year is not None and year < filters["year_min"]:
        return False
    if "year_max" in filters and year is not None and year > filters["year_max"]:
        return False

    if "evidence_level_max" in filters:
        ev = meta.get("evidence_level")
        if ev is not None and ev > filters["evidence_level_max"]:
            return False

    return True
