"""
Cross-encoder reranking via Cohere Rerank API.

After hybrid fusion returns the top-20 candidates, the reranker scores
each (query, chunk) pair with a cross-encoder model trained on relevance
judgements.  This is the most expensive retrieval step per query (~$0.001
for 20 candidates) but typically yields the largest quality improvement.

Model: `rerank-english-v3.0` (Cohere).
Fallback: if `COHERE_API_KEY` is empty, skip reranking and return the
fusion results directly — the system degrades gracefully.

Study-design weighting:
    After obtaining Cohere relevance scores, they are combined with the
    chunk's `study_design_weight` (from STUDY_DESIGN_WEIGHTS) via a
    weighted geometric mean:

        final_score = rerank_score^0.8 * study_design_weight^0.2

    This gives mild preference to higher-evidence-grade sources at equal
    semantic relevance without overriding relevance for low-evidence chunks
    that directly answer the question.

Public API (to be implemented):
    class Reranker:
        def __init__(self, cohere_client, settings: Settings): ...

        def rerank(
            self,
            query: str,
            candidates: list[SearchResult],
            top_k: int | None = None,
        ) -> list[SearchResult]:
            Return `top_k` results sorted by final_score descending.
            Each result's `score` field is updated to `final_score`.

        def is_available(self) -> bool:
            Return True if Cohere API key is configured.
"""
