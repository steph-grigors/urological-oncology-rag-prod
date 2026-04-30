"""
BM25 sparse keyword search.

Provides a complementary retrieval signal to dense vector search.
BM25 excels at exact clinical term matching — drug names, gene symbols,
numeric thresholds — that semantic embeddings sometimes miss.

Implementation approach:
    - At query time, queries the BM25 index stored in Postgres full-text
      search (tsvector columns on the `chunks` table) using `ts_rank_cd`.
    - Uses medical stop-word list to avoid penalising domain vocabulary.
    - Parameters BM25_K1 and BM25_B from `config/constants.py` are passed
      as Postgres GUC variables via `SET LOCAL` within the query transaction.
    - Returns ranked `SearchResult` objects with the same schema as
      `vector_search.py` for seamless fusion in `hybrid.py`.
    - Supports the same `SearchFilter` as `VectorSearch` for consistent
      metadata pre-filtering.

Public API (to be implemented):
    class BM25Search:
        def __init__(self, db_session, settings: Settings): ...

        def search(
            self,
            query: str,
            top_k: int,
            filter: SearchFilter | None = None,
        ) -> list[SearchResult]: ...

        def index_chunks(self, chunks: list[Chunk]) -> None:
            # Called by the ingestion pipeline to populate the Postgres
            # full-text index alongside the Qdrant upsert.
            ...
"""
