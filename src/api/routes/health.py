"""
GET /health — liveness, readiness, and info probes.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Request

from src.api.middleware.auth import require_api_key

router = APIRouter(prefix="/health", tags=["health"])

_startup_time = time.time()


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
    ok = True

    # Qdrant check
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is not None:
        try:
            retriever._store.collection_stats()
            checks["qdrant"] = "ok"
        except Exception as exc:
            checks["qdrant"] = f"error: {exc}"
            ok = False
    else:
        checks["qdrant"] = "not_configured"

    # Postgres / AuditLogger check
    audit_logger = getattr(request.app.state, "audit_logger", None)
    if audit_logger is not None:
        try:
            from sqlalchemy import text
            with audit_logger.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            checks["postgres"] = "ok"
        except Exception as exc:
            checks["postgres"] = f"error: {exc}"
            ok = False
    else:
        checks["postgres"] = "not_configured"

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
