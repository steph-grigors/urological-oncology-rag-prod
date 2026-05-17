"""
Cross-encoder reranking via Cohere Rerank API.

After RRF fusion produces the top-20 candidates, the reranker scores each
(query, chunk) pair with a cross-encoder that outperforms bi-encoder
cosine similarity for relevance judgements.

Final score formula (study-design + recency weighted):
    final_score = relevance_score^0.70 * study_design_weight^0.15 * recency_weight^0.15

Weights sum to 1.0 (weighted geometric mean). Relevance dominates; study
design and recency provide a tie-breaking nudge without overriding a directly
relevant chunk from an older or lower-evidence source.

Recency tiers (years before current year → weight):
    0–2  → 1.00   3–5  → 0.90   6–10 → 0.80   11–15 → 0.70   >15 → 0.60

RCT landmark boost:
    Chunks with study_design = "rct" published within the last 2 years receive
    an additional 1.10× multiplier on top of their recency weight, surfacing
    recent landmark trials (e.g. EV-302/NIAGARA NEJM 2024) above older protocols
    for the same indication. The cap of 1.10× prevents this from overriding a
    directly relevant older RCT (LATITUDE 2017, VISION 2021 remain dominant when
    the Cohere relevance score strongly favours them).

Graceful degradation:
    If COHERE_API_KEY is empty or the API call fails, chunks are returned
    as RankedChunk objects with relevance_score = original RRF score.
"""

from __future__ import annotations

import datetime
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
                recency_weight = _recency_weight(chunk)
                rct_boost = _rct_landmark_boost(chunk)
                final_score = rel ** 0.70 * design_weight ** 0.15 * (recency_weight * rct_boost) ** 0.15
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


def _recency_weight(chunk: ScoredChunk) -> float:
    """Return a weight in [0.60, 1.00] based on how recent the paper is."""
    year = chunk.metadata.get("year")
    if not year:
        return 0.75  # unknown year: neutral
    try:
        age = datetime.datetime.now().year - int(year)
    except (ValueError, TypeError):
        return 0.75
    if age <= 2:
        return 1.00
    if age <= 5:
        return 0.90
    if age <= 10:
        return 0.80
    if age <= 15:
        return 0.70
    return 0.60


def _rct_landmark_boost(chunk: ScoredChunk) -> float:
    """Return 1.10 for RCTs published within 2 years, 1.0 otherwise.

    Surfaces recent landmark trials (e.g. EV-302 NEJM 2024) above older
    protocols for the same indication without overriding the Cohere relevance
    score for older RCTs that score highly on direct relevance.
    """
    design = chunk.metadata.get("study_design", "unknown")
    if design != "rct":
        return 1.0
    year = chunk.metadata.get("year")
    if not year:
        return 1.0
    try:
        age = datetime.datetime.now().year - int(year)
    except (ValueError, TypeError):
        return 1.0
    return 1.10 if age <= 2 else 1.0


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
