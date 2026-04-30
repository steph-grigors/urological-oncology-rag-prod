"""
Postgres document store abstraction.

Postgres serves three roles in this system:
    1. Full-text search index (tsvector) for BM25-style keyword retrieval.
    2. Persistent store for the audit log (`audit_log` table).
    3. Metadata store for papers and chunks, enabling queries like
       "how many papers indexed per topic" without hitting Qdrant.

All database access uses SQLAlchemy 2.0 async sessions to avoid blocking
the FastAPI event loop.

Migrations are managed by Alembic (`db/migrations/`).  Never alter the
schema by hand — add a migration instead.

Public API (to be implemented):
    class DocumentStore:
        def __init__(self, settings: Settings): ...

        async def upsert_chunks(self, chunks: list[Chunk]) -> None:
            Insert or update chunk rows and refresh tsvector columns.

        async def full_text_search(
            self,
            query: str,
            top_k: int,
            filter: SearchFilter | None = None,
        ) -> list[BM25Result]:
            Execute ts_rank_cd query and return ranked results.

        async def get_chunk(self, chunk_id: str) -> Chunk | None: ...

        async def write_audit(self, record: AuditRecord) -> None: ...

        async def get_corpus_stats(self) -> CorpusStats:
            Return counts per topic, year, study_design for dashboard.
"""
