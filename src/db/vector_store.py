"""
Qdrant client abstraction.

Wraps qdrant-client with a domain-specific interface so the rest of the
codebase never imports from qdrant-client directly.  Enables future
migration to a different vector store without touching retrieval logic.

Collection configuration:
    - Vector size: EMBEDDING_DIMENSION (1536 for text-embedding-3-small)
    - Distance: Cosine
    - HNSW index: m=16, ef_construct=100 (production-grade recall vs. speed)
    - Quantisation: none (scalar quantisation optional for memory savings)
    - Payload index on `topic`, `year`, `study_design` for filtered search

Payload schema stored per point:
    pmc_id, pmid, doi, title, authors, journal, year, topic,
    section_name, section_type, study_design, chunk_index, total_chunks,
    cancer_subtype, patient_population, intervention

Public API (to be implemented):
    class QdrantStore:
        def __init__(self, settings: Settings): ...

        def upsert(self, points: list[QdrantPoint]) -> None: ...
        def search(
            self,
            query_vector: list[float],
            top_k: int,
            filter: dict | None = None,
        ) -> list[ScoredPoint]: ...
        def delete(self, ids: list[str]) -> None: ...
        def count(self) -> int: ...
        def ensure_collection(self) -> None: ...
"""
