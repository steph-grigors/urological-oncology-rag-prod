# Local Architecture

## Overview

```
rag_ui.py (Streamlit)          localhost:8501
    │  HTTP POST /query
    ▼
FastAPI (src/api/)             localhost:8000
    │
    ├── Qdrant (Docker)        localhost:6333   685K vectors
    ├── PostgreSQL (Docker)    localhost:5432   audit logs, conversations
    └── OpenAI / Cohere API    (outbound HTTPS)
```

The UI has zero business logic — it is a pure HTTP client that POSTs to the API
and renders the JSON response. All intelligence lives in the API layer.

---

## Layer Breakdown

| Layer | Location | Role |
|---|---|---|
| **UI** | `rag_ui.py` | Streamlit frontend. Calls `POST /query`, renders answer + source cards |
| **API** | `src/api/` | FastAPI. Orchestrates retrieval → rerank → generate |
| **Retrieval** | `src/retrieval/` | Dense (Qdrant) + BM25 hybrid search, Cohere reranker, RRF fusion |
| **Generation** | `src/generation/` | GPT-4o-mini prompt, confidence gating, clinical safety prefix |
| **Vector DB** | Qdrant (Docker volume `qdrant_storage`) | 685K embedded chunks, cosine similarity search |
| **Document DB** | PostgreSQL (Docker volume `postgres_data`) | Audit logs, conversation history |
| **Ingestion** | `src/ingestion/` | One-time pipeline: PMC fetch → parse → chunk → embed → upsert |
| **Evaluation** | `src/evaluation/` | Golden set + heuristic judges for offline quality regression |

---

## Request Flow

```
Doctor types query
      │
      ▼ POST /query  {query, cancer_types, top_k}
FastAPI route (src/api/routes/query.py)
      │
      ├─ Auth check (X-API-Key header)
      │
      ├─ Embed query  →  OpenAI text-embedding-3-small (1536 dims)
      │
      ├─ Dense search  →  Qdrant (top_k_retrieval=20)
      ├─ BM25 search   →  in-memory index (top_k_retrieval=20)
      ├─ RRF fusion    →  merge + deduplicate
      │
      ├─ Rerank  →  Cohere rerank-english-v3.0 (top_k_rerank=5)
      │
      ├─ Confidence gate  →  score chunks; route to hedged/refuse if low
      │
      ├─ Generate  →  GPT-4o-mini with clinical system prompt
      │
      └─ Return JSON  {answer, sources, evidence_quality, latency_ms, ...}
```

---

## Key Files

```
rag_ui.py                          Streamlit UI
config/settings.py                 All env-var settings (pydantic-settings)
docker/docker-compose.yml          Full stack definition
docker/.env                        API keys and service URLs
src/api/main.py                    FastAPI app factory + lifespan
src/api/routes/query.py            POST /query endpoint
src/api/routes/health.py           GET /health/live and /health/ready
src/api/routes/eval.py             POST /eval/run, GET /eval/status
src/api/routes/ingestion.py        GET /ingestion/status
src/api/middleware/auth.py         X-API-Key validation
src/retrieval/retriever.py         Hybrid search orchestrator
src/retrieval/reranker.py          Cohere reranker + recency weighting
src/retrieval/bm25_search.py       In-memory BM25 index
src/generation/generator.py        ClinicalGenerator (LLM call)
src/generation/prompts.py          System prompt + temporal conflict rules
src/generation/confidence.py       Confidence gate (hedged / refuse)
src/ingestion/pipeline.py          Ingestion CLI entry point
src/ingestion/fetch.py             NCBI E-utilities search + fetch
src/ingestion/chunk.py             Section-aware chunker
src/ingestion/embed.py             OpenAI embeddings + Qdrant upsert
```

---

## Docker Services

| Service | Image | Port | Data |
|---|---|---|---|
| `api` | Custom (Dockerfile) | 8000 | Stateless |
| `ui` | Custom (Dockerfile) | 8501 | Stateless |
| `qdrant` | qdrant/qdrant:v1.17.0 | 6333, 6334 | `qdrant_storage` volume |
| `postgres` | postgres:16-alpine | 5432 | `postgres_data` volume |
| `ingestion-cron` | mcuadros/ofelia | — | Runs Sundays 02:00 UTC |
| `langfuse` | langfuse/langfuse:2 | 3000 | Optional profile |

---

## Start Commands

```bash
# Start core stack (API + Qdrant + PostgreSQL)
docker compose -f docker/docker-compose.yml --env-file docker/.env up api qdrant postgres

# Start with UI
docker compose -f docker/docker-compose.yml --env-file docker/.env up

# Start with weekly ingestion cron
docker compose -f docker/docker-compose.yml --env-file docker/.env --profile ingestion-cron up

# Run ingestion pipeline directly (host shell)
python -m src.ingestion.pipeline --mode full --limit 20000 --since-date 2010/01/01

# Run UI only (local dev, API already running)
streamlit run rag_ui.py
```

---

## Environment Variables (docker/.env)

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Embeddings (text-embedding-3-small) + generation (GPT-4o-mini) |
| `COHERE_API_KEY` | Reranking (rerank-english-v3.0) |
| `QDRANT_URL` | `http://localhost:6333` (host) or `http://qdrant:6333` (Docker internal) |
| `QDRANT_COLLECTION` | `urological_oncology_papers` |
| `POSTGRES_URL` | PostgreSQL connection string |
| `API_KEYS` | Comma-separated or JSON array of valid access keys |
| `APP_ENV` | `development` (auth bypass) / `production` |
| `NCBI_API_KEY` | Optional — raises NCBI rate limit from 3 to 10 req/s |
