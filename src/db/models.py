"""
SQLAlchemy ORM models.

Defines the Postgres schema for all application tables.
Use `alembic revision --autogenerate` after adding or modifying models.

Tables (to be implemented):

    papers
        pmc_id (PK), pmid, doi, title, abstract, journal, year, topic,
        authors (JSONB), study_design, cancer_subtype, patient_population,
        intervention, comparator, created_at, updated_at

    chunks
        id (PK, deterministic hash), pmc_id (FK → papers.pmc_id),
        text, section_name, section_type, chunk_index, total_chunks,
        tsvector_col (TSVECTOR, generated from text, indexed GIN),
        created_at

    audit_log
        query_id (PK, UUID), timestamp (timestamptz, NOT NULL),
        question (TEXT), rewritten_query (TEXT), answer (TEXT),
        confidence (FLOAT), gate_decision (VARCHAR), model (VARCHAR),
        provider (VARCHAR), input_tokens (INT), output_tokens (INT),
        latency_ms (FLOAT), sources (JSONB), user_id (VARCHAR),
        session_id (VARCHAR), hallucinated_citations (JSONB),
        -- INSERT-only: no UPDATE/DELETE granted to app role

Indexes:
    chunks.tsvector_col     GIN index for full-text search
    chunks.pmc_id           B-tree for join to papers
    audit_log.timestamp     B-tree for time-range queries
    audit_log.session_id    B-tree for conversation history retrieval
"""
