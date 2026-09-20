"""
GET /health — liveness, readiness, and info probes.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Request

from src.api.middleware.auth import require_api_key
from src.observability.logging import get_logger

router = APIRouter(prefix="/health", tags=["health"])
logger = get_logger(__name__)

_startup_time = time.time()


def _check_qdrant(retriever) -> tuple[str, bool]:
    """Return (status, ok) for the vector store.

    A reachable-but-empty collection is reported as a failure rather than as
    "ok". QdrantStore.ensure_collection creates the collection when it is
    missing, so a typo in QDRANT_COLLECTION produces a brand-new empty one and
    every downstream call succeeds: collection_stats() returns cleanly, the
    probe went green, and the API served an empty corpus. Retrieval would
    return nothing, every query would fall through to the parametric-knowledge
    path, and nothing anywhere would say why.
    """
    if retriever is None:
        return "not_configured", True
    try:
        stats = retriever._store.collection_stats()
    except Exception:
        logger.warning("Qdrant health check failed", exc_info=True)
        return "error", False

    point_count = stats.get("point_count")
    if not point_count:
        logger.error(
            "Qdrant collection %r is reachable but empty — check QDRANT_COLLECTION",
            stats.get("collection"),
        )
        return "empty", False
    return "ok", True


def _check_postgres(audit_logger) -> tuple[str, bool]:
    """Return (status, ok) for the audit-log database."""
    if audit_logger is None:
        return "not_configured", True
    try:
        from sqlalchemy import text

        with audit_logger.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        # Detail goes to the log, not the response: these probes are
        # unauthenticated, and a driver exception routinely embeds the host,
        # port, database name or credentials from the DSN.
        logger.warning("Postgres health check failed", exc_info=True)
        return "error", False
    return "ok", True


@router.get("")
async def health_check(request: Request) -> Any:
    """Combined health check: Qdrant + Postgres + OpenAI key configured.

    Returns {status: ok|degraded, checks: {qdrant, postgres, openai}}.
    Suitable for load-balancer probes and the docker-compose healthcheck.
    """
    from fastapi.responses import JSONResponse

    checks: dict[str, str] = {}
    ok = True

    checks["qdrant"], qdrant_ok = _check_qdrant(
        getattr(request.app.state, "retriever", None)
    )
    checks["postgres"], postgres_ok = _check_postgres(
        getattr(request.app.state, "audit_logger", None)
    )
    ok = qdrant_ok and postgres_ok

    settings = getattr(request.app.state, "settings", None)
    if settings and settings.openai_api_key:
        checks["openai"] = "ok"
    else:
        checks["openai"] = "not_configured"

    return JSONResponse(
        content={"status": "ok" if ok else "degraded", "checks": checks},
        status_code=200 if ok else 503,
    )


@router.get("/live")
async def liveness() -> dict[str, str]:
    """Always 200 — confirms the process is running."""
    return {"status": "ok"}


@router.get("/ready")
async def readiness(request: Request) -> dict[str, Any]:
    """
    Returns 200 only when all required dependencies respond.
    Returns 503 with a failure map when any dependency is down.
    """
    from fastapi.responses import JSONResponse

    checks: dict[str, str] = {}

    checks["qdrant"], qdrant_ok = _check_qdrant(
        getattr(request.app.state, "retriever", None)
    )
    checks["postgres"], postgres_ok = _check_postgres(
        getattr(request.app.state, "audit_logger", None)
    )
    ok = qdrant_ok and postgres_ok

    status_code = 200 if ok else 503
    return JSONResponse(
        content={"status": "ok" if ok else "degraded", "checks": checks},
        status_code=status_code,
    )


@router.get("/info")
async def info(
    request: Request,
    _api_key: str = Depends(require_api_key),
) -> dict[str, Any]:
    """System metadata — requires API key auth."""
    settings = getattr(request.app.state, "settings", None)
    retriever = getattr(request.app.state, "retriever", None)
    collection_count = None
    if retriever is not None:
        try:
            collection_count = retriever._store.collection_stats().get("point_count")
        except Exception:
            pass

    return {
        "app_env": settings.app_env if settings else "unknown",
        "generation_model": settings.generation_model if settings else "unknown",
        "embedding_model": settings.embedding_model if settings else "unknown",
        "collection_count": collection_count,
        "uptime_seconds": round(time.time() - _startup_time, 1),
    }
