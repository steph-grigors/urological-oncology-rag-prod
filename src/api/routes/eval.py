"""
POST /eval — offline evaluation trigger endpoints.
"""

from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.middleware.auth import require_api_key

router = APIRouter(prefix="/eval", tags=["eval"])

# In-memory store for demo purposes — replace with DB-backed store in production
_runs: dict[str, dict] = {}


class EvalRunRequest(BaseModel):
    mode: Literal["full", "quick", "regression"] = "quick"
    golden_set_version: str | None = None
    baseline_run_id: str | None = None


@router.post("/run")
async def trigger_eval(
    body: EvalRunRequest,
    _api_key: str = Depends(require_api_key),
) -> dict:
    """Accept an evaluation run request and return a run_id."""
    if body.mode == "regression" and not body.baseline_run_id:
        raise HTTPException(
            status_code=422, detail="baseline_run_id required for regression mode"
        )
    run_id = str(uuid.uuid4())
    _runs[run_id] = {"status": "running", "mode": body.mode}
    return {"run_id": run_id, "status": "accepted"}


@router.get("/status/{run_id}")
async def eval_status(
    run_id: str,
    _api_key: str = Depends(require_api_key),
) -> dict:
    """Poll an evaluation run by ID."""
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/results/latest")
async def latest_results(_api_key: str = Depends(require_api_key)) -> dict:
    """Return the most recent completed evaluation report."""
    completed = [r for r in _runs.values() if r.get("status") == "completed"]
    if not completed:
        raise HTTPException(status_code=404, detail="No completed runs")
    return completed[-1]


@router.get("/results")
async def all_results(_api_key: str = Depends(require_api_key)) -> list:
    """Return all evaluation run summaries."""
    return list(_runs.values())
