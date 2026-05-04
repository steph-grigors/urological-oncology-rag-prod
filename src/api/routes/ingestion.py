"""Ingestion status endpoint."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/ingestion", tags=["ingestion"])

_PROGRESS_PATH = Path("data/ingestion_progress.json")


@router.get("/status", summary="Current ingestion run progress")
async def ingestion_status() -> JSONResponse:
    """Return the progress of the running or most recently completed ingestion pipeline.

    The file is written atomically after every 50-paper batch, so this endpoint
    always reflects a consistent snapshot even during a live run.
    """
    if not _PROGRESS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No ingestion run found. Run the pipeline first: python -m src.ingestion.pipeline",
        )
    try:
        data = json.loads(_PROGRESS_PATH.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read progress file: {exc}")
    return JSONResponse(content=data)
