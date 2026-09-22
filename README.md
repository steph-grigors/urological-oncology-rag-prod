# Urological Oncology RAG

A production-grade Retrieval-Augmented Generation system for evidence-based answers about urological oncology. Covers prostate, bladder, kidney, testicular, penile and adrenal cancer across 28,438 full-text papers from PubMed Central (2010–2026), indexed as 713,883 section-aware chunks.

Two endpoints: `POST /query` for clinical evidence questions, and `POST /treatment-card` for structured treatment cards built from patient data, intended as support for multidisciplinary team meetings rather than as a diagnostic.

Corpus figures measured against the production collection on 2026-09-22.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Docker Compose Stack                        │
│                                                                     │
│  ┌──────────────┐      ┌──────────────────────────────────────────┐ │
│  │  Streamlit   │─────▶│             FastAPI  :8000               │ │
│  │  UI  :8501   │◀─────│                                          │ │
│  └──────────────┘      │  POST /query        POST /treatment-card │ │
│                        │  POST /eval/run     GET /eval/results    │ │
│                        │  GET  /eval/status  GET /docs            │ │
│                        └────────────┬─────────────────────────────┘ │
│                                     │                               │
│               ┌─────────────────────┼──────────────────┐           │
│               ▼                     ▼                  ▼           │
│  ┌─────────────────┐  ┌─────────────────────┐  ┌──────────────┐   │
│  │  Qdrant  :6333  │  │  PostgreSQL  :5432  │  │  Langfuse    │   │
│  │  vectors + all  │  │  audit log,         │  │  :3000       │   │
│  │  chunk metadata │  │  conversations      │  │  (optional   │   │
│  └─────────────────┘  └─────────────────────┘  │   profile)   │   │
│                                                 └──────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

Query path:
  User question
    → Hybrid retrieval (dense cosine + sparse BM25, top-20 candidates)
    → Cohere cross-encoder reranking (top-5)
    → Anthropic claude-sonnet-4-6 generation with citation grounding
    → Evidence quality classification (high / hedged / caveated / insufficient)
    → Langfuse trace (latency, token cost, quality scores)
```

---

## Quick Start

### Prerequisites

- Docker ≥ 24 and Docker Compose ≥ 2.20
- OpenAI API key (embeddings — required)
- Anthropic API key (generation — required)
- Cohere API key (cross-encoder reranking — optional; retrieval falls back to
  reciprocal-rank-fusion order without it)
- Python 3.11 only if you want to run the ingestion pipeline or the test suite
  outside Docker

### 1. Clone and configure

```bash
git clone https://github.com/steph-grigors/urological-oncology-rag-prod.git
cd urological-oncology-rag-prod

cp docker/.env.example docker/.env
# Edit docker/.env — fill in OPENAI_API_KEY and ANTHROPIC_API_KEY at minimum
```

`docker/.env.example` lists every variable the application reads, with safe
defaults filled in and every credential left blank. `docker/.env` itself is
gitignored and never committed.

Before exposing the stack beyond localhost, change the Postgres credentials in
both `docker/.env` and `docker/docker-compose.yml`; they default to `rag:rag`.

### 2. Start services

```bash
# Standard (api + qdrant + postgres + ui)
docker-compose --env-file docker/.env -f docker/docker-compose.yml up

# With self-hosted Langfuse tracing
docker-compose --env-file docker/.env -f docker/docker-compose.yml --profile langfuse up

# API only (no UI)
docker-compose --env-file docker/.env -f docker/docker-compose.yml up api qdrant postgres
```

Services start in dependency order. The API waits for Qdrant and Postgres health checks before accepting traffic.

### 3. Check health

```bash
curl http://localhost:8000/health
# {"status":"ok","checks":{"qdrant":"ok","postgres":"ok","openai":"ok"}}
```

### 4. Run your first query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"query": "What are the first-line treatment options for muscle-invasive bladder cancer?"}'
```

Open the Streamlit UI at **http://localhost:8501**.

### 5. Run the tests

The test suite needs no API keys, no network and no running services: external
clients are mocked and Qdrant runs in memory.

```bash
pip install -r requirements-dev.txt   # installs requirements.txt too
pytest                                # unit + integration + eval
pytest tests/unit                     # unit only
```

`requirements-dev.txt` is separate from `requirements.txt` so the runtime image
carries no test dependencies.

---

## Ingestion Pipeline

The ingestion pipeline fetches papers from PubMed Central, chunks them, generates embeddings, and loads them into Qdrant and Postgres.

### Steps

The pipeline is a single command. Fetch, parse, quality-gate, chunk, extract
metadata, embed and upsert all run in one pass, checkpointing every 50 papers.

```bash
# Everything since a date, all six cancer types
python -m src.ingestion.pipeline --since-date 2026/05/10 --limit 1000

# The last week — what the weekly scheduled job runs
python -m src.ingestion.pipeline --since-days 7 --limit 2000

# One topic, see what would be fetched without fetching it
python -m src.ingestion.pipeline --cancer-types penile --dry-run

# After any ingestion run, refresh the BM25 disk cache so the next
# API restart loads in seconds rather than re-scrolling the collection
python scripts/rebuild_bm25_cache.py
```

There is no separate fetch, embed or load step to run. Earlier versions of this
README documented `src.ingestion.load` and `src.ingestion.run_pipeline`; neither
module exists.

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--cancer-types` | all six | Cancer types to fetch |
| `--limit` | 300 | Papers per topic |
| `--chunk-size` | 200 words | Words per chunk (env: `CHUNK_SIZE_WORDS`) |
| `--chunk-overlap` | 30 words | Overlap between chunks (env: `CHUNK_OVERLAP_WORDS`) |
| `--collection` | `urological_oncology_papers` | Qdrant collection name |

Ingestion state is checkpointed to `data/ingestion_state.json` — interrupted runs resume from the last completed paper.

---

## Running Tests

```bash
# Install dev dependencies
pip install -r requirements.txt

# Unit tests only (no external services required)
pytest tests/unit/ -v

# Integration tests (requires running Qdrant + Postgres)
pytest tests/integration/ -v

# Full suite with coverage
pytest --cov=src --cov-report=term-missing

# Fast subset (markers)
pytest -m "not slow" -v
```

Test configuration lives in [pytest.ini](pytest.ini). Integration tests read `QDRANT_URL` and `POSTGRES_URL` from the environment; defaults target `localhost`.

---

## Evaluation Suite

The eval suite scores the system on a golden question set using five metrics: faithfulness, answer relevance, context precision, context recall, and clinical safety.

### Trigger via API (requires `ADMIN_API_KEY`)

```bash
# Start a quick run (10 questions)
curl -X POST http://localhost:8000/eval/run \
  -H "X-API-Key: your-admin-key-here" \
  -H "Content-Type: application/json" \
  -d '{"mode": "quick"}'
# Returns: {"run_id": "...", "status": "accepted"}

# Poll status
curl http://localhost:8000/eval/status/<run_id> \
  -H "X-API-Key: your-admin-key-here"

# Fetch latest completed report
curl http://localhost:8000/eval/results/latest \
  -H "X-API-Key: your-api-key-here"
```

### Run modes

| Mode | Questions | Use case |
|------|-----------|----------|
| `quick` | 10 | Pre-deploy smoke test |
| `full` | All (~100) | Scheduled nightly run |
| `regression` | All | Compare against a `baseline_run_id` |

### Run directly (no API)

```bash
python -m src.evaluation.runner --mode quick --output-dir data/evaluation
```

Results are written to `data/evaluation/latest_metrics.json` and timestamped JSON files.

---

## Environment Variables

All variables are documented in [docker/.env.example](docker/.env.example). Required variables are marked below.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI key for embeddings (`text-embedding-3-small`) |
| `ANTHROPIC_API_KEY` | Yes* | — | Anthropic key for generation; *required if `GENERATION_PROVIDER=anthropic` |
| `GENERATION_PROVIDER` | No | `anthropic` | `anthropic` or `openai` |
| `GENERATION_MODEL` | No | `claude-sonnet-4-6` | Full model ID passed to the provider |
| `COHERE_API_KEY` | No | — | Enables cross-encoder reranking; omit to skip |
| `API_KEYS` | No | — | Comma-separated accepted API keys; auth bypassed in `development` when empty |
| `ADMIN_API_KEY` | No | — | Key for `POST /eval/run`; falls back to `API_KEYS` when empty |
| `APP_ENV` | No | `development` | `development` / `staging` / `production` |
| `LOG_LEVEL` | No | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `RATE_LIMIT_PER_MINUTE` | No | `60` | Requests per minute per IP; `0` disables |
| `QDRANT_URL` | No | `http://qdrant:6333` | Qdrant endpoint |
| `QDRANT_API_KEY` | No | — | Qdrant auth token (Community edition has none) |
| `QDRANT_COLLECTION` | No | `urological_oncology_papers` | Collection name |
| `POSTGRES_URL` | No | `postgresql+asyncpg://rag:rag@postgres:5432/rag` | asyncpg connection string |
| `LANGFUSE_PUBLIC_KEY` | No | — | Langfuse public key; all three required to enable tracing |
| `LANGFUSE_SECRET_KEY` | No | — | Langfuse secret key |
| `LANGFUSE_HOST` | No | `https://cloud.langfuse.com` | Langfuse endpoint (or `http://langfuse:3000` for self-hosted) |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Must be 1536-dim to match collection schema |
| `METADATA_EXTRACTION_MODEL` | No | `gpt-4o-mini` | Study-design extraction during ingestion |
| `TOP_K_RETRIEVAL` | No | `20` | Dense + sparse candidates before reranking |
| `TOP_K_RERANK` | No | `5` | Chunks passed to the LLM after reranking |
| `CONFIDENCE_THRESHOLD` | No | `0.45` | Minimum reranker score to produce a non-refused answer |
| `MAX_CONTEXT_CHARS` | No | `6000` | Max combined context length passed to the LLM |
| `CHUNK_SIZE_WORDS` | No | `200` | Words per chunk during ingestion |
| `CHUNK_OVERLAP_WORDS` | No | `30` | Overlap words between adjacent chunks |
| `API_BACKEND_URL` | No | `http://api:8000` | Backend URL as seen by the Streamlit container |

---

## API Endpoints

Base URL: `http://localhost:8000`

Interactive docs available at `/docs` (hidden in `production` mode).

### POST /query

Submit a question and receive a grounded answer.

**Headers:** `X-API-Key: <key>`

**Request body:**

```json
{
  "query": "What are the side effects of enzalutamide?",
  "cancer_types": [],
  "top_k": 5,
  "conversation_id": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | The clinical question |
| `cancer_types` | string[] | Filter by topic; empty = all |
| `top_k` | int | Sources to return (1–20) |
| `conversation_id` | string? | UUID for multi-turn context; omit for single-turn |

**Response:**

```json
{
  "answer": "Enzalutamide is associated with ...",
  "sources": [
    {
      "chunk_id": "pmid-12345678-abstract",
      "title": "Enzalutamide in metastatic CRPC",
      "authors": ["Smith J", "Lee K"],
      "journal": "J Clin Oncol",
      "year": 2022,
      "study_design": "RCT",
      "sample_size": 1199,
      "section": "Results",
      "key_finding": "Enzalutamide reduced risk of radiographic progression by 71%.",
      "pmid": "12345678"
    }
  ],
  "evidence_quality": "high",
  "confidence_score": 0.82,
  "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "latency_ms": {
    "retrieval": 45,
    "rerank": 120,
    "generation": 1840,
    "total": 2010
  }
}
```

`evidence_quality` values: `high` · `hedged` · `caveated` · `insufficient`

---

### GET /health

Aggregate health across Qdrant, Postgres, and OpenAI key presence.

```json
{"status": "ok", "checks": {"qdrant": "ok", "postgres": "ok", "openai": "ok"}}
```

Returns `200 ok` or `503 degraded`.

---

### GET /health/live

Kubernetes liveness probe — returns `200` if the process is alive.

### GET /health/ready

Kubernetes readiness probe — returns `200` when the retriever and generator are initialised.

---

### POST /eval/run

Trigger an evaluation run in the background. **Requires `ADMIN_API_KEY`.**

```json
{"mode": "quick"}
```

Returns `{"run_id": "<uuid>", "status": "accepted"}` immediately.

### GET /eval/status/{run_id}

Poll run progress. Returns `running`, `completed`, or `failed`.

### GET /eval/results/latest

Latest completed evaluation report with aggregate scores.

### GET /eval/results

All run summaries for the current server lifetime.

---

## Project Structure

```
.
├── config/
│   └── settings.py          # Pydantic BaseSettings — all env vars
├── docker/
│   ├── Dockerfile           # Multi-stage builder + runtime
│   ├── docker-compose.yml   # All services
│   └── .env.example         # Template — copy to docker/.env
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app, lifespan startup
│   │   ├── middleware/      # Auth (HMAC key comparison), rate limit
│   │   └── routes/          # query, health, eval
│   ├── generation/          # LLM client (Anthropic + OpenAI), prompt templates
│   ├── ingestion/           # PubMed fetch, chunking, embedding, load
│   ├── retrieval/           # Hybrid search (Qdrant dense + BM25), reranking
│   ├── evaluation/          # Golden set, judges, runner
│   └── observability/       # Langfuse tracing helpers
├── tests/
│   ├── unit/
│   └── integration/
├── rag_ui.py                # Streamlit frontend
└── requirements.txt
```
