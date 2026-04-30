"""
POST /eval/run — trigger an offline evaluation pass.

This endpoint is for operators and CI pipelines.  It is not exposed to
end users and requires the API key auth middleware.

Endpoints:
    POST /eval/run
        Body (EvalRunRequest):
            mode: "full" | "quick" | "regression"   default "quick"
            golden_set_version: str | None           default "latest"
            baseline_run_id: str | None              required for "regression" mode
        Response: { "run_id": str, "status": "accepted" }
        The evaluation runs in a background task; poll /eval/status/{run_id}
        for results.

    GET /eval/status/{run_id}
        Returns EvaluationReport if completed, or {"status": "running"} if
        still in progress.

    GET /eval/results/latest
        Returns the most recent completed EvaluationReport.

    GET /eval/results
        Returns a list of all completed EvaluationReport summaries (id, date,
        overall score) for the dashboard's historical performance chart.
"""
