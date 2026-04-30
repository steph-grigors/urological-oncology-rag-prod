"""
Dense vector search against Qdrant.

Wraps the Qdrant client with a narrow interface tailored to this project's
data model. All callers should use this module rather than the Qdrant SDK
directly, so the underlying store can be swapped without changing retrieval
logic.

Features:
- Embeds the query with the same model used at index time
  (controlled by `settings.embedding_model`).
- Supports optional metadata filtering (topic, year range, study_design)
  via Qdrant's filter DSL, translated from a typed `SearchFilter` object.
- Returns a ranked list of `SearchResult` objects with distance scores
  normalised to [0, 1].
- Caches query embeddings in an in-process LRU cache (maxsize=1000) to
  avoid redundant API calls within a session.

Public API (to be implemented):
    class VectorSearch:
        def __init__(self, qdrant_client, openai_client, settings: Settings): ...

        def search(
            self,
            query: str,
            top_k: int,
            filter: SearchFilter | None = None,
        ) -> list[SearchResult]: ...

    SearchFilter(dataclass)
        topics: list[str] | None
        year_min: int | None
        year_max: int | None
        study_designs: list[str] | None

    SearchResult(dataclass)
        chunk_id: str
        text: str
        metadata: dict
        score: float          # cosine similarity, higher is better
        rank: int
"""
