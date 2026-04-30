"""
Langfuse distributed tracing integration.

Every query is traced as a Langfuse `Trace` with nested `Span` objects for
each pipeline step.  This provides per-step latency breakdown, token counts,
and cost attribution visible in the Langfuse dashboard.

Trace structure for a typical query:
    Trace: query (query_id)
        Span: retrieval
            Span: vector_search         (latency, top_k)
            Span: bm25_search           (latency, top_k)
            Span: rrf_fusion            (latency, num_candidates)
            Span: reranking             (latency, cohere_tokens)
        Span: confidence_gating         (score, gate_decision)
        Span: generation                (latency, input_tokens, output_tokens, model)
        Span: citation_check            (hallucinated_citations)

Graceful degradation:
    If LANGFUSE_PUBLIC_KEY is not set, all tracing calls become no-ops.
    Application code should never have try/except blocks around tracer calls —
    the tracer itself handles the disabled state transparently.

Public API (to be implemented):
    def setup_tracing(settings: Settings) -> None:
        Initialise the Langfuse client.  Call once at startup.

    @contextmanager
    def trace_query(query_id: str, question: str) -> Iterator[QueryTrace]:
        Context manager that opens a Langfuse trace and yields a
        `QueryTrace` helper for creating child spans.

    class QueryTrace:
        def span(self, name: str, **kwargs) -> ContextManager[Span]: ...
        def score(self, name: str, value: float, comment: str = "") -> None: ...
        def set_metadata(self, **kwargs) -> None: ...
"""
