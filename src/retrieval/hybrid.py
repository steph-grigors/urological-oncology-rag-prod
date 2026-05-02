"""
Reciprocal Rank Fusion (RRF) of dense and BM25 retrieval results.

RRF formula per chunk:
    score = 1 / (k + rank_dense) + 1 / (k + rank_bm25)

where k = 60 (default) and rank is 1-based.  A chunk absent from one list
gets rank = top_k + 1 (one position beyond the last ranked chunk in that
list) rather than infinity, so single-list chunks still receive a partial
score.

References: Cormack, Clarke & Buettcher (2009). "Reciprocal Rank Fusion
outperforms Condorcet and individual rank learning methods."
"""

from __future__ import annotations

from src.db.vector_store import ScoredChunk

_DEFAULT_K = 60


def rrf_fusion(
    dense_results: list[ScoredChunk],
    bm25_results: list[ScoredChunk],
    k: int = _DEFAULT_K,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """
    Merge dense and BM25 result lists via RRF.

    Parameters
    ----------
    dense_results : ranked dense search results (index 0 = best)
    bm25_results  : ranked BM25 search results (index 0 = best)
    k             : RRF smoothing constant (default 60)
    top_k         : truncate output to this many chunks; None = return all

    Returns
    -------
    Merged list sorted by RRF score descending, with each chunk's `score`
    field set to the RRF score.  The original dense/BM25 scores are not
    preserved (use metadata if you need them downstream).
    """
    # 1-based rank maps: chunk_id → rank
    dense_rank: dict[str, int] = {
        c.chunk_id: i + 1 for i, c in enumerate(dense_results)
    }
    bm25_rank: dict[str, int] = {
        c.chunk_id: i + 1 for i, c in enumerate(bm25_results)
    }

    # Fallback rank for chunks missing from one list
    fallback = max(len(dense_results), len(bm25_results), 1) + 1

    # Union of all chunk IDs; keep a lookup for text + metadata
    chunks_by_id: dict[str, ScoredChunk] = {}
    for c in dense_results:
        chunks_by_id[c.chunk_id] = c
    for c in bm25_results:
        if c.chunk_id not in chunks_by_id:
            chunks_by_id[c.chunk_id] = c

    scored: list[ScoredChunk] = []
    for chunk_id, chunk in chunks_by_id.items():
        r_dense = dense_rank.get(chunk_id, fallback)
        r_bm25 = bm25_rank.get(chunk_id, fallback)
        rrf_score = 1.0 / (k + r_dense) + 1.0 / (k + r_bm25)
        scored.append(ScoredChunk(
            chunk_id=chunk_id,
            text=chunk.text,
            score=rrf_score,
            metadata=chunk.metadata,
        ))

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k] if top_k is not None else scored
