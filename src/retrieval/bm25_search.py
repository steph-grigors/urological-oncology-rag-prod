"""
In-memory BM25 keyword search using the rank_bm25 library.

BM25 complements dense vector search by excelling at exact clinical term
matching (drug names, gene symbols, numeric thresholds) that embeddings
sometimes miss or mis-rank.

Usage:
    # Build from a pre-loaded list of ScoredChunk objects
    bm25 = BM25Search(chunks)

    # Or load the entire Qdrant collection at startup
    bm25 = BM25Search.from_qdrant(store)

    results = bm25.search("enzalutamide progression-free survival", top_k=20)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi

from src.db.vector_store import ScoredChunk

if TYPE_CHECKING:
    from src.db.vector_store import QdrantStore


# ── Tokeniser (shared with sparse vector helpers) ─────────────────────────────

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if w]


# ── BM25Search ────────────────────────────────────────────────────────────────

class BM25Search:
    """
    In-memory BM25 index over all indexed chunks.

    The index is built once at startup and held in memory.  For a 41 K-chunk
    corpus the index is ~50–100 MB — well within typical server RAM.
    Re-index after large ingestion runs by calling `build(chunks)` again.
    """

    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self._chunks: list[ScoredChunk] = []
        self._bm25: BM25Okapi | None = None
        if chunks:
            self.build(chunks)

    @classmethod
    def from_qdrant(cls, store: "QdrantStore") -> "BM25Search":
        """Scroll the full Qdrant collection and build the index."""
        chunks = store.scroll_all()
        return cls(chunks)

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
