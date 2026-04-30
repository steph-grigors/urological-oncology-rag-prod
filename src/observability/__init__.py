"""
Observability package.

Provides structured logging, distributed tracing (Langfuse), and an
immutable audit log for every query processed by the system.

Import order:
    logging.setup_logging()   — call once at application startup
    audit.AuditLogger         — injected into the query route handler
    tracing.tracer            — used as a context manager around pipeline steps
"""
