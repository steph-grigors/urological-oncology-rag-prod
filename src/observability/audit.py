"""
Immutable audit log — every query and answer persisted to Postgres.

Clinical decision-support tools must maintain an audit trail for
regulatory and quality-assurance purposes.  This module writes a record
to the `audit_log` table (defined in `db/models.py`) for every query that
completes, whether or not the system returned an answer.

Each audit record captures:
    query_id          UUID, primary key
    timestamp         UTC, immutable once written
    question          raw user query text
    rewritten_query   standalone query after context expansion (if chat mode)
    answer            full generated answer (or refusal text)
    confidence        scalar confidence score
    gate_decision     "high" | "hedged" | "caveated" | "refused"
    model             generation model used
    provider          "anthropic" | "openai"
    input_tokens      LLM input token count
    output_tokens     LLM output token count
    latency_ms        total pipeline latency
    sources           JSON array of {pmid, title, section, score}
    user_id           optional identifier from auth middleware
    session_id        optional conversation session ID
    hallucinated_cites list of citation numbers that were stripped

Immutability:
    Records are INSERT-only; no UPDATE or DELETE is ever issued by the
    application.  Row-level security in Postgres prevents deletion by the
    application role.

Public API (to be implemented):
    class AuditLogger:
        def __init__(self, db_session): ...

        async def log(
            self,
            query_id: str,
            question: str,
            result: GenerationResult,
            retrieval_result: RetrievalResult,
            confidence: float,
            gate: ConfidenceGate,
            user_id: str | None = None,
            session_id: str | None = None,
        ) -> None: ...
"""
