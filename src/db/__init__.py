"""
Database package.

Provides client abstractions for:
    - Qdrant   (vector store)   → vector_store.QdrantStore
    - Postgres (document store, BM25 index, audit log) → document_store.DocumentStore

SQLAlchemy models are defined in `models.py`.
Alembic migrations live in `migrations/`.
"""
