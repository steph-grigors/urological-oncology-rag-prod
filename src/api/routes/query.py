"""
POST /query — main RAG query endpoint.

Request body (QueryRequest):
    question: str                   required, 10–500 chars
    topic_filter: str | None        optional — "prostate"|"bladder"|"kidney"|"testicular"
    year_min: int | None            optional publication year filter
    year_max: int | None
    study_design_filter: list[str] | None
    top_k: int | None               override TOP_K_RERANK for this request
    model: str | None               override generation model
    session_id: str | None          for multi-turn conversation tracking
    stream: bool                    default False

Response body (QueryResponse):
    query_id: str                   UUID for audit tracing
    question: str
    answer: str
    confidence: float
    gate: str                       "high" | "hedged" | "caveated" | "refused"
    sources: list[SourceDoc]
    latency_ms: dict[str, float]    per-step breakdown
    model: str
    provider: str
    rewritten_query: str | None     populated if session_id provided

SourceDoc:
    pmid: str | None
    doi: str | None
    title: str
    section: str
    year: str
    topic: str
    study_design: str | None
    score: float
    text_preview: str               first 300 chars of the chunk

Error responses:
    422  Validation error
    429  Rate limit exceeded (from rate_limit middleware)
    401  Invalid or missing API key (from auth middleware)
    503  Upstream service unavailable (Qdrant or LLM API down)

Streaming:
    When stream=True, returns a Server-Sent Events (SSE) stream.
    The final SSE event is a JSON object matching QueryResponse.
"""
