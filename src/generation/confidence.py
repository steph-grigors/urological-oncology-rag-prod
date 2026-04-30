"""
Confidence gating — decides whether to answer, hedge, or refuse.

A clinical decision-support system must know when it doesn't know enough.
This module computes a confidence signal from the retrieval result and
determines the appropriate response posture.

Confidence signal sources (combined into a scalar in [0, 1]):
    1. mean_rerank_score    — average cross-encoder score across top-k chunks
    2. score_spread         — std dev of rerank scores (high spread = uncertain)
    3. topic_coverage       — fraction of retrieved chunks whose topic matches
                              the detected query topic (cross-topic leakage penalty)
    4. source_diversity     — number of distinct PMC IDs in top-k
                              (single-paper answers are less reliable)

Gating logic:
    confidence >= CONFIDENCE_HIGH   → answer normally
    CONFIDENCE_LOW <= c < HIGH      → prepend HEDGED_ANSWER_PREFIX
    CONFIDENCE_REFUSE <= c < LOW    → answer but with strong caveat and
                                      "consult primary literature" instruction
    confidence < CONFIDENCE_REFUSE  → hard refusal, no answer produced

Public API (to be implemented):
    def compute_confidence(retrieval_result: RetrievalResult) -> float:
        Return a scalar confidence score in [0, 1].

    def gate(
        confidence: float,
        settings: Settings,
    ) -> ConfidenceGate:
        Return a ConfidenceGate enum value for the generation layer.

    class ConfidenceGate(str, Enum):
        HIGH = "high"
        HEDGED = "hedged"
        CAVEATED = "caveated"
        REFUSED = "refused"

    def confidence_to_metadata(confidence: float) -> dict:
        Return a dict of confidence sub-scores for audit logging.
"""
