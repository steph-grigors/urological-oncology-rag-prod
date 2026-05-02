"""
Confidence gating — decides whether to answer, hedge, or refuse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from config.constants import CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_REFUSE

if TYPE_CHECKING:
    from src.retrieval.reranker import RankedChunk


class ConfidenceGate(str, Enum):
    HIGH = "high"
    HEDGED = "hedged"
    CAVEATED = "caveated"
    REFUSED = "refused"


@dataclass
class ConfidenceResult:
    score: float
    sufficient: bool
    reason: str


def compute_confidence(
    chunks: list["RankedChunk"],
    query_cancer_types: list[str] | None = None,
) -> ConfidenceResult:
    """Return a ConfidenceResult with a scalar score in [0, 1]."""
    if not chunks:
        return ConfidenceResult(score=0.0, sufficient=False, reason="no_chunks")

    top3 = [c.relevance_score for c in chunks[:3]]
    base_score = sum(top3) / len(top3)

    adj = 0.0
    reasons: list[str] = []

    # Evidence quality: boost for RCT/meta-analysis (level ≤ 2), penalise for review/unknown (level ≥ 5)
    levels = [c.metadata.get("evidence_level", 6) for c in chunks]
    if any(lv <= 2 for lv in levels):
        adj += 0.1
        reasons.append("evidence_boost")
    elif all(lv >= 5 for lv in levels):
        adj -= 0.1
        reasons.append("evidence_penalty")

    # Single-paper penalty: multiple chunks from one source are less diverse
    pmcids = {c.metadata.get("pmcid") for c in chunks if c.metadata.get("pmcid")}
    if len(pmcids) == 1 and len(chunks) > 1:
        adj -= 0.1
        reasons.append("single_paper_penalty")

    # Score spread penalty: high variance signals uncertain retrieval
    all_scores = [c.relevance_score for c in chunks]
    if len(all_scores) > 1:
        mean_s = sum(all_scores) / len(all_scores)
        variance = sum((s - mean_s) ** 2 for s in all_scores) / len(all_scores)
        if math.sqrt(variance) > 0.3:
            adj -= 0.05
            reasons.append("spread_penalty")

    # Topic mismatch penalty: retrieved chunks are from a different cancer type
    if query_cancer_types:
        chunk_types: set[str] = set()
        for c in chunks:
            ct = c.metadata.get("cancer_type", [])
            if isinstance(ct, str):
                ct = [ct]
            chunk_types.update(ct)
        if not chunk_types.intersection(set(query_cancer_types)):
            adj -= 0.1
            reasons.append("topic_mismatch_penalty")

    score = max(0.0, min(1.0, base_score + adj))
    return ConfidenceResult(
        score=score,
        sufficient=score >= CONFIDENCE_LOW,
        reason="; ".join(reasons) if reasons else "nominal",
    )


def gate(score: float) -> ConfidenceGate:
    """Map a scalar confidence score to a response posture."""
    if score >= CONFIDENCE_HIGH:
        return ConfidenceGate.HIGH
    if score >= CONFIDENCE_LOW:
        return ConfidenceGate.HEDGED
    if score >= CONFIDENCE_REFUSE:
        return ConfidenceGate.CAVEATED
    return ConfidenceGate.REFUSED


def confidence_to_metadata(result: ConfidenceResult) -> dict:
    """Return a flat dict of confidence sub-scores for audit logging."""
    return {
        "score": result.score,
        "gate": gate(result.score).value,
        "sufficient": result.sufficient,
        "reason": result.reason,
    }
