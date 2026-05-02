"""
Cross-encoder reranking via Cohere Rerank API.

After RRF fusion produces the top-20 candidates, the reranker scores each
(query, chunk) pair with a cross-encoder that outperforms bi-encoder
cosine similarity for relevance judgements.

Final score formula (study-design weighted):
    final_score = relevance_score^0.8 * study_design_weight^0.2

This gives mild preference to higher-evidence sources at equal semantic
relevance without overriding a directly relevant lower-evidence chunk.

Graceful degradation:
    If COHERE_API_KEY is empty or the API call fails, chunks are returned
    as RankedChunk objects with relevance_score = original RRF score.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cohere

from config.constants import STUDY_DESIGN_WEIGHTS
from src.db.vector_store import ScoredChunk

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "rerank-english-v3.0"


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class RankedChunk:
    """A reranked chunk carrying both the final weighted score and the raw
    Cohere relevance score."""
    chunk_id: str
    text: str
    score: float            # final weighted score (or RRF score if no Cohere)
    relevance_score: float  # raw Cohere score (0–1); equals score if fallback
    metadata: dict = field(default_factory=dict)


# ── CohereReranker ────────────────────────────────────────────────────────────

class CohereReranker:
    """
    Wraps the Cohere Rerank endpoint.

    Pass api_key="" to disable reranking (chunks are returned with their
    RRF scores preserved in both `score` and `relevance_score`).
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._model = model
        self._client = cohere.ClientV2(api_key=api_key) if api_key else None

    def is_available(self) -> bool:
        return self._client is not None

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        """
        Rerank `chunks` for `query` and return the top `top_n`.

        If Cohere is unavailable, returns the first `top_n` chunks unchanged
        (preserving the RRF ordering) with relevance_score = original score.
        """
        if not chunks:
            return []

        if not self.is_available():
            return _passthrough(chunks, top_n)

        t0 = time.perf_counter()
        try:
            response = self._client.rerank(  # type: ignore[union-attr]
                model=self._model,
                query=query,
                documents=[c.text for c in chunks],
                top_n=top_n,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.info("Cohere rerank latency: %.1f ms", latency_ms)

            ranked: list[RankedChunk] = []
            for result in response.results:
                chunk = chunks[result.index]
                rel = float(result.relevance_score)
                design_weight = _design_weight(chunk)
                final_score = rel ** 0.8 * design_weight ** 0.2
                ranked.append(RankedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=final_score,
                    relevance_score=rel,
                    metadata=chunk.metadata,
                ))
            return ranked

        except Exception:
            logger.exception("Cohere rerank failed — falling back to RRF order")
            return _passthrough(chunks, top_n)


# ── Private helpers ───────────────────────────────────────────────────────────

def _design_weight(chunk: ScoredChunk) -> float:
    design = chunk.metadata.get("study_design", "unknown")
    # constants.py uses verbose names; ingestion uses short codes — check both
    return STUDY_DESIGN_WEIGHTS.get(design, STUDY_DESIGN_WEIGHTS.get("unknown", 0.5))


def _passthrough(chunks: list[ScoredChunk], top_n: int) -> list[RankedChunk]:
    return [
        RankedChunk(
            chunk_id=c.chunk_id,
            text=c.text,
            score=c.score,
            relevance_score=c.score,
            metadata=c.metadata,
        )
        for c in chunks[:top_n]
    ]
