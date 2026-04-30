"""
FastAPI application factory and startup/shutdown lifecycle.

`create_app()` is the entry point — it instantiates all shared resources
(database clients, retriever, generator) and attaches them to the app's
`state` object so route handlers can access them via dependency injection.

Startup sequence:
    1. Load settings (get_settings())
    2. setup_logging(settings.log_level)
    3. setup_tracing(settings)
    4. Connect to Qdrant and Postgres (with health checks)
    5. Instantiate VectorSearch, BM25Search, HybridSearch, Reranker
    6. Instantiate Retriever, Generator, AuditLogger
    7. Register routers: /query, /health, /eval

Shutdown:
    Close Qdrant and Postgres connections gracefully.

CORS:
    Restricted to same-origin in production; configurable via ALLOWED_ORIGINS
    env var for local development.

To run locally:
    uvicorn src.api.main:app --reload --port 8000
"""
