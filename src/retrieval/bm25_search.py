"""
In-memory BM25 keyword search using the bm25s library.

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
    The index is persisted to BM25_CACHE_DIR (default data/bm25_cache/).
    Restarts load in seconds instead of scrolling 685K chunks from Qdrant.
    Call save() after (re-)building to update it.
    The cache is invalidated automatically when the Qdrant point count changes.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING

import bm25s
import numpy as np

from src.db.vector_store import ScoredChunk

if TYPE_CHECKING:
    from src.db.vector_store import QdrantStore

logger = logging.getLogger(__name__)

BM25_CACHE_DIR = "data/bm25_cache"


# ── Tokeniser (shared with sparse vector helpers) ─────────────────────────────

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if w]


# ── BM25Search ────────────────────────────────────────────────────────────────

class BM25Search:
    """
    In-memory BM25 index over all indexed chunks.

    The index is built once at startup and held in memory.  The index is
    persisted to disk (BM25_CACHE_DIR) so restarts load in seconds instead
    of re-scrolling Qdrant.  The cache is invalidated when the Qdrant point
    count no longer matches the cached count.
    """

    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self._chunks: list[ScoredChunk] = []
        self._bm25: bm25s.BM25 | None = None
        if chunks:
            self.build(chunks)

    @classmethod
    def from_qdrant(
        cls,
        store: "QdrantStore",
        cache_dir: str = BM25_CACHE_DIR,
    ) -> "BM25Search":
        """Load BM25 from disk cache if valid, otherwise scroll Qdrant and save."""
        live_count = store.count()
        cached = cls._load_cache(cache_dir, live_count)
        if cached is not None:
            logger.info("BM25 loaded from cache (%d chunks)", cached.size)
            return cached

        logger.info("BM25 cache miss (live=%d) — scrolling Qdrant…", live_count)
        chunks = store.scroll_all()
        instance = cls(chunks)
        instance.save(cache_dir, live_count)
        return instance

    # ── Disk cache ────────────────────────────────────────────────────────

    def save(self, cache_dir: str = BM25_CACHE_DIR, count: int | None = None) -> None:
        """Persist the index to disk. count is stored for invalidation."""
        try:
            path = Path(cache_dir)
            path.mkdir(parents=True, exist_ok=True)
            self._bm25.save(str(path), show_progress=False)
            with (path / "chunks.pkl").open("wb") as f:
                pickle.dump(self._chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
            meta = {"count": count if count is not None else len(self._chunks)}
            (path / "meta.json").write_text(json.dumps(meta))
            logger.info("BM25 cache saved to %s (%d chunks)", cache_dir, len(self._chunks))
        except Exception as exc:
            logger.warning("Could not save BM25 cache: %s", exc)

    @classmethod
    def _load_cache(cls, cache_dir: str, live_count: int) -> "BM25Search | None":
        path = Path(cache_dir)
        meta_path = path / "meta.json"
        chunks_path = path / "chunks.pkl"
        if not (path.exists() and meta_path.exists() and chunks_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("count") != live_count:
                logger.info(
                    "BM25 cache stale (cached=%d, live=%d)",
                    meta.get("count"), live_count,
                )
                return None
            bm25_index = bm25s.BM25.load(str(path), load_corpus=False)
            with chunks_path.open("rb") as f:
                chunks = pickle.load(f)
            instance = cls.__new__(cls)
            instance._chunks = chunks
            instance._bm25 = bm25_index
            return instance
        except Exception as exc:
            logger.warning("Could not load BM25 cache: %s", exc)
            return None

    # ── Index management ──────────────────────────────────────────────────

    def build(self, chunks: list[ScoredChunk]) -> None:
        """(Re-)build the BM25 index from a chunk list."""
        self._chunks = list(chunks)
        corpus_tokens = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = bm25s.BM25(method="robertson")
        self._bm25.index(corpus_tokens, show_progress=False)

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

        scores: np.ndarray = self._bm25.get_scores(query_tokens)

        # score == 0.0 means the document contains none of the query tokens.
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
