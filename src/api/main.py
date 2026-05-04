"""
FastAPI application factory and startup/shutdown lifecycle.

To run locally:
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import get_settings
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.routes import eval as eval_router
from src.api.routes import health, ingestion as ingestion_router, query
from src.observability.logging import get_logger, request_id_var, setup_logging
from src.observability.tracing import setup_tracing

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    setup_tracing(settings)

    app.state.settings = settings

    # ── Qdrant + retrieval stack ───────────────────────────────────────────
    try:
        from qdrant_client import QdrantClient

        from src.db.vector_store import QdrantStore
        from src.generation.generator import ClinicalGenerator
        from src.generation.llm_client import LLMClient
        from src.retrieval.bm25_search import BM25Search
        from src.retrieval.reranker import CohereReranker
        from src.retrieval.retriever import RAGRetriever

        qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        store = QdrantStore(qdrant_client, collection_name=settings.qdrant_collection)

        bm25 = BM25Search.from_qdrant(store)
        reranker = CohereReranker(api_key=settings.cohere_api_key)

        from openai import OpenAI
        openai_client = OpenAI(api_key=settings.openai_api_key)

        retriever = RAGRetriever(
            store=store,
            bm25=bm25,
            reranker=reranker,
            openai_client=openai_client,
            embedding_model=settings.embedding_model,
            top_k_retrieval=settings.top_k_retrieval,
            top_k_rerank=settings.top_k_rerank,
        )
        app.state.retriever = retriever
        logger.info("Retrieval stack initialised")
    except Exception as exc:
        logger.warning("Retrieval stack not available: %s", exc)
        app.state.retriever = None

    # ── LLM generator ──────────────────────────────────────────────────────
    try:
        from src.generation.generator import ClinicalGenerator
        from src.generation.llm_client import LLMClient

        llm_client = LLMClient(
            provider=settings.generation_provider,
            model=settings.generation_model,
            api_key=(
                settings.anthropic_api_key
                if settings.generation_provider == "anthropic"
                else settings.openai_api_key
            ),
        )
        app.state.generator = ClinicalGenerator(llm_client=llm_client)
        logger.info("Generator initialised (provider=%s)", settings.generation_provider)
    except Exception as exc:
        logger.warning("Generator not available: %s", exc)
        app.state.generator = None

    # ── Audit logger ───────────────────────────────────────────────────────
    try:
        from src.observability.audit import AuditLogger
        app.state.audit_logger = AuditLogger(settings.postgres_url)
        logger.info("AuditLogger initialised")
    except Exception as exc:
        logger.warning("AuditLogger not available: %s", exc)
        app.state.audit_logger = None

    # ── Document store (conversation history, corpus stats) ────────────────
    try:
        from src.db.document_store import DocumentStore
        app.state.document_store = DocumentStore(settings.postgres_url)
        logger.info("DocumentStore initialised")
    except Exception as exc:
        logger.warning("DocumentStore not available: %s", exc)
        app.state.document_store = None

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("Application shutting down")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Urological Oncology RAG API",
        version="1.0.0",
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    # CORS
    allowed_origins = ["*"] if settings.app_env == "development" else []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Rate limiting
    app.add_middleware(RateLimitMiddleware, settings=settings)

    # Request-ID middleware (inject UUID per request, add to response header)
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = rid
        request_id_var.set(rid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    # Global exception handler — never expose tracebacks to clients
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred"},
        )

    # Routers
    app.include_router(query.router)
    app.include_router(health.router)
    app.include_router(eval_router.router)
    app.include_router(ingestion_router.router)

    return app


app = create_app()
