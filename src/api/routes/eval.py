"""
/eval — offline evaluation trigger endpoints.

POST /eval/run        trigger a background eval run (admin key required)
GET  /eval/status/:id poll run status
GET  /eval/results/latest latest completed report
GET  /eval/results    all run summaries
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel

from src.api.middleware.auth import require_admin_api_key, require_api_key

router = APIRouter(prefix="/eval", tags=["eval"])

# In-memory run registry — restarts clear history (acceptable for portfolio project)
_runs: dict[str, dict] = {}


class EvalRunRequest(BaseModel):
    mode: Literal["full", "quick", "regression"] = "quick"
    baseline_run_id: str | None = None


# ── POST /eval/run ─────────────────────────────────────────────────────────────

@router.post("/run")
async def trigger_eval(
    body: EvalRunRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    _: str = Depends(require_admin_api_key),
) -> dict[str, str]:
    """Trigger an evaluation run asynchronously.

    Requires the ADMIN_API_KEY header.  Returns immediately with a run_id
    that can be polled via GET /eval/status/{run_id}.
    """
    if body.mode == "regression" and not body.baseline_run_id:
        raise HTTPException(
            status_code=422,
            detail="baseline_run_id is required for regression mode",
        )

    retriever = getattr(request.app.state, "retriever", None)
    generator = getattr(request.app.state, "generator", None)

    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialised — cannot run evaluation",
        )

    run_id = str(uuid.uuid4())
    _runs[run_id] = {"status": "running", "mode": body.mode, "run_id": run_id}

    background_tasks.add_task(
        _run_eval_background,
        run_id=run_id,
        mode=body.mode,
        retriever=retriever,
        generator=generator,
    )
    return {"run_id": run_id, "status": "accepted"}


# ── Background worker ──────────────────────────────────────────────────────────

async def _run_eval_background(
    run_id: str,
    mode: str,
    retriever: Any,
    generator: Any,
) -> None:
    from src.evaluation.runner import run_evaluation

    loop = asyncio.get_event_loop()
    try:
        report = await loop.run_in_executor(
            None,
            lambda: run_evaluation(
                retriever=retriever,
                mode=mode,
                generator=generator,
                output_dir="data/evaluation",
            ),
        )
        _runs[run_id] = {
            "status": "completed",
            "run_id": run_id,
            "mode": mode,
            "timestamp": report.timestamp,
            "golden_set_version": report.golden_set_version,
            "total_queries": report.total_queries,
            "aggregate": {
                "faithfulness": round(report.aggregate.faithfulness, 4),
                "answer_relevance": round(report.aggregate.answer_relevance, 4),
                "context_precision": round(report.aggregate.context_precision, 4),
                "context_recall": round(report.aggregate.context_recall, 4),
                "clinical_safety": round(report.aggregate.clinical_safety, 4),
                "overall": round(report.aggregate.overall, 4),
            },
            "latency_stats": report.latency_stats,
            "passed_regression": report.passed_regression,
        }
    except Exception as exc:
        _runs[run_id] = {
            "status": "failed",
            "run_id": run_id,
            "error": str(exc),
        }


# ── GET /eval/status/{run_id} ──────────────────────────────────────────────────

@router.get("/status/{run_id}")
async def eval_status(
    run_id: str,
    _: str = Depends(require_admin_api_key),
) -> dict:
    """Poll an evaluation run by ID."""
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


# ── GET /eval/results/latest ───────────────────────────────────────────────────

@router.get("/results/latest")
async def latest_results(
    _: str = Depends(require_api_key),
) -> dict:
    """Return the most recent completed evaluation report."""
    completed = [r for r in _runs.values() if r.get("status") == "completed"]
    if not completed:
        # Fall back to the file written by a previous run
        try:
            import json
            from pathlib import Path

            path = Path("data/evaluation/latest_metrics.json")
            if path.exists():
                with open(path) as fh:
                    return json.load(fh)
        except Exception:
            pass
        raise HTTPException(status_code=404, detail="No completed runs found")
    return completed[-1]


# ── GET /eval/results ──────────────────────────────────────────────────────────

@router.get("/results")
async def all_results(
    _: str = Depends(require_api_key),
) -> list:
    """Return all evaluation run summaries for this server lifetime."""
    return list(_runs.values())
