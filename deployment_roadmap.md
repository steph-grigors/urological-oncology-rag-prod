# Deployment Roadmap

## Entry Points

```
docker-compose --env-file docker/.env -f docker/docker-compose.yml up
```

| Service | Command | URL |
|---------|---------|-----|
| API | `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2` | http://localhost:8000 |
| UI | `streamlit run rag_ui.py --server.port 8501` | http://localhost:8501 |

---

## Step 1 — Configure secrets

```bash
cp docker/.env.example docker/.env
```

Edit `docker/.env` and fill in:

| Variable | Required | Notes |
|----------|----------|-------|
| `OPENAI_API_KEY` | Yes | Embeddings (`text-embedding-3-small`) |
| `ANTHROPIC_API_KEY` | Yes | Generation (`claude-sonnet-4-6`) |
| `API_KEYS` | Yes | Comma-separated keys accepted by `/query` |
| `ADMIN_API_KEY` | Yes | Key accepted by `POST /eval/run` |
| `COHERE_API_KEY` | Recommended | Enables cross-encoder reranking; without it the system falls back to RRF ordering (lower retrieval quality) |
| `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` | Optional | Enables per-query distributed tracing |

---

## Step 2 — Start infrastructure services

```bash
docker-compose --env-file docker/.env -f docker/docker-compose.yml up qdrant postgres -d
```

Wait for both to pass their health checks before proceeding. You can verify:

```bash
docker ps  # both should show "healthy"
```

---

## Step 3 — Apply Postgres schema

Run the Alembic migration to create all five tables (`papers`, `chunks`, `audit_log`, `api_keys`, `conversation_history`):

```bash
# Temporarily point at localhost (Postgres is port-forwarded from docker)
POSTGRES_URL="postgresql://rag:rag@localhost:5432/rag" alembic upgrade head
```

Expected output: `Running upgrade -> 001, initial schema`

---

## Step 4 — Run the ingestion pipeline

The raw papers are already in `data/papers_fulltext/` and processed chunks in `data/processed_fulltext/`. This step embeds and loads them into Qdrant and Postgres.

```bash
# Incremental mode — picks up from data/ without re-fetching from PMC
python -m src.ingestion.pipeline --mode incremental

# Full mode — re-fetches from PubMed Central + re-embeds (expensive: ~$2-5 in OpenAI costs)
python -m src.ingestion.pipeline --mode full
```

The pipeline checkpoints progress to `data/ingestion_state.json`. If interrupted, re-run the same command and it will resume from the last completed paper.

---

## Step 5 — Verify data loaded

```bash
# Check all dependencies are reachable
curl http://localhost:8000/health
# Expected: {"status":"ok","checks":{"qdrant":"ok","postgres":"ok","openai":"ok"}}

# Check collection stats (requires API key)
curl http://localhost:8000/health/info -H "X-API-Key: <your-api-key>"
```

---

## Step 6 — Start the full stack

```bash
# Standard (api + qdrant + postgres + ui)
docker-compose --env-file docker/.env -f docker/docker-compose.yml up

# With self-hosted Langfuse tracing
docker-compose --env-file docker/.env -f docker/docker-compose.yml --profile langfuse up
```

---

## Step 7 — Smoke test

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{"query": "What is the first-line treatment for muscle-invasive bladder cancer?", "top_k": 5}'
```

Check the response contains:
- `"answer"` — non-empty text with `[Doc N]` citations
- `"sources"` — list of SourceCard objects (non-empty)
- `"evidence_quality"` — one of `high | hedged | caveated | insufficient`
- `"latency_ms"` — breakdown with `retrieval`, `rerank`, `generation`, `total`

Open the Streamlit UI at **http://localhost:8501** and run the same query through the interface.

---

## Step 8 — Run the evaluation baseline

```bash
# Trigger a quick run (10 questions from the golden set)
curl -X POST http://localhost:8000/eval/run \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-admin-key>" \
  -d '{"mode": "quick"}'
# Returns: {"run_id": "<uuid>", "status": "accepted"}

# Poll until complete
curl http://localhost:8000/eval/status/<run_id> \
  -H "X-API-Key: <your-admin-key>"

# Fetch the results
curl http://localhost:8000/eval/results/latest \
  -H "X-API-Key: <your-api-key>"
```

Save this result — it becomes the regression baseline for future deploys. Target scores:

| Metric | Minimum threshold |
|--------|------------------|
| `faithfulness` | ≥ 0.80 |
| `answer_relevance` | ≥ 0.75 |
| `context_precision` | ≥ 0.70 |
| `clinical_safety` | ≥ 0.95 |
| `overall` | ≥ 0.80 |

---

## Step 9 — Conversation history ✅ (done)

Already implemented. The `/query` endpoint now:
1. Fetches the last 10 messages (5 turns) from the `conversation_history` table when `conversation_id` is provided
2. Passes them to the generator as context
3. Persists the new user + assistant turn after generation

No action required.

---

## Step 10 — Pre-production hardening

Before sharing the deployment with anyone else:

| Item | Priority | What to do |
|------|----------|-----------|
| **Rate limiter** | Medium | The current in-process sliding-window dict is per-process and resets on restart. Fine for a single-container test; replace with Redis (`redis-py` + `limits` library) before running multiple API replicas. |
| **Streamlit auth** | Medium | The 2-query free-tier limit is session-state only — anyone can bypass it by refreshing. Set `APP_ENV=production` in `docker/.env` to hide `/docs`. Add backend API key enforcement to the UI if it will be public-facing. |
| **Secrets management** | High (shared deploy) | Move from `.env` file to a secrets manager (AWS Secrets Manager, GCP Secret Manager, Docker Secrets) before any deployment shared with others. |
| **Log level** | Low | Set `LOG_LEVEL=WARNING` in `docker/.env` for production. `INFO` is noisy under real load. |
| **CORS** | High (public API) | `allowed_origins = ["*"]` in development mode. Set `APP_ENV=production` and configure explicit origins before any public exposure. |
| **BM25 index startup time** | Low | The BM25 index is rebuilt from Qdrant on every startup (full collection scroll). For 41K chunks this is acceptable; for larger corpora consider persisting the index to disk with `joblib`. |

---

## Remaining known gaps

| Gap | Impact | Notes |
|-----|--------|-------|
| `rank_bm25` not in `requirements.txt` | High — BM25 search will fail at startup | Add `rank_bm25>=0.2.2` to `requirements.txt` before building the Docker image |
| Rate limiter not Redis-backed | Medium — multi-instance deployments lose per-IP state | Single container: not a blocker |
| No golden set expansion | Low | `tests/fixtures/golden_queries.json` has minimal fixture queries; add real clinical questions before running a `full` eval |
