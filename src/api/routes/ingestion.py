"""Ingestion status endpoint."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from src.api.middleware.auth import require_api_key
from src.observability.logging import get_logger

router = APIRouter(prefix="/ingestion", tags=["ingestion"])
logger = get_logger(__name__)

_PROGRESS_PATH = Path("data/ingestion_progress.json")


@router.get("/status", summary="Current ingestion run progress")
async def ingestion_status(
    _api_key: str = Depends(require_api_key),
) -> JSONResponse:
    """Return the progress of the running or most recently completed ingestion pipeline.

    The file is written atomically after every 50-paper batch, so this endpoint
    always reflects a consistent snapshot even during a live run.
    """
    if not _PROGRESS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No ingestion run found.",
        )
    try:
        data = json.loads(_PROGRESS_PATH.read_text())
    except Exception:
        logger.warning("Could not read ingestion progress file", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not read progress file")
    return JSONResponse(content=data)
