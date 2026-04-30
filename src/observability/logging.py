"""
Structured JSON logging configuration.

All application log output uses JSON format so it can be ingested by log
aggregation platforms (Datadog, Loki, CloudWatch) without parsing.

Log record fields (added via a custom `logging.Filter`):
    timestamp   ISO-8601 UTC
    level       DEBUG / INFO / WARNING / ERROR / CRITICAL
    logger      module path (e.g., "src.retrieval.retriever")
    message     human-readable message
    request_id  UUID from the incoming HTTP request (if available)
    query_id    UUID assigned to each RAG query for trace correlation
    latency_ms  for log records emitted at query completion

Sensitive field scrubbing:
    API keys, patient identifiers, and raw query text are scrubbed from
    WARNING+ log levels before emission using PII_SCRUB_PATTERNS from
    constants.py.  DEBUG logs are unredacted (for local development only).

Public API (to be implemented):
    def setup_logging(level: str = "INFO") -> None:
        Configure the root logger with the JSON formatter.
        Call once at application startup (in `api/main.py`).

    def get_logger(name: str) -> logging.Logger:
        Return a pre-configured logger.  Use instead of `logging.getLogger`.

    class RequestContextFilter(logging.Filter):
        Injects request_id and query_id from contextvars into each record.
"""
