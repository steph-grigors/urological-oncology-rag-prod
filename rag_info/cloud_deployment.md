# Cloud Deployment

## Target Architecture

```
Doctors' browsers
      │  HTTPS
      ▼
HuggingFace Spaces  (optional — Streamlit UI only)
      │  HTTPS → API_BACKEND_URL
      ▼
Cloud VPS  (Hetzner / DigitalOcean / AWS)
├── FastAPI          :8000   (behind Caddy/Nginx reverse proxy)
├── Streamlit UI     :8501   (optional if not using HF Spaces)
├── Qdrant           :6333   (internal only — not exposed publicly)
├── PostgreSQL       :5432   (internal only — not exposed publicly)
└── Ofelia cron               (weekly incremental ingestion, Sundays 02:00 UTC)
      │
      └── outbound HTTPS → OpenAI, Cohere, NCBI
```

---

## Server Requirements

| Component | RAM |
|---|---|
| Qdrant vectors (685K × 1536 dims × 4 bytes) | ~4.2 GB |
| Qdrant HNSW index overhead (~2.5×) | ~10.5 GB total |
| BM25 tokenized corpus (685K chunks) | ~1.6 GB |
| FastAPI + OS + headroom | ~3 GB |
| **Total** | **~15 GB → 32 GB server recommended** |

CX32 (16 GB) is too tight once Qdrant's HNSW index is fully loaded. Use CX42.

| Provider | Instance | RAM | CPU | Storage | Price |
|---|---|---|---|---|---|
| **Hetzner** (recommended) | **CX42** | **32 GB** | **8 vCPU** | 240 GB SSD | ~€26/month |
| DigitalOcean | 32 GB Droplet | 32 GB | 8 vCPU | 640 GB SSD | ~$192/month |
| AWS | t3.2xlarge | 32 GB | 8 vCPU | EBS | ~$240/month |

---

## Deployment Steps

### 1 — Provision the server

```bash
# On Hetzner: create CX32 Ubuntu 22.04 server
# SSH in, install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

### 2 — Copy the project

```bash
# From your local machine
git push origin master          # push latest code
ssh user@your-server-ip "git clone https://github.com/steph-grigors/urological-oncology-rag-prod.git"
```

### 3 — Migrate Qdrant data (avoids re-embedding ~$3 cost)

```bash
# On your local machine — create a snapshot
curl -X POST http://localhost:6333/collections/urological_oncology_papers/snapshots

# Download the snapshot file (name returned by the above)
curl -O http://localhost:6333/collections/urological_oncology_papers/snapshots/<snapshot-name>

# Upload to server
scp <snapshot-name> user@your-server-ip:~/

# On the server — restore into the running Qdrant container
docker compose -f docker/docker-compose.yml --env-file docker/.env up -d qdrant
curl -X POST "http://localhost:6333/collections/urological_oncology_papers/snapshots/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "snapshot=@<snapshot-name>"
```

### 4 — Configure environment

```bash
# On the server — copy your .env and set production values
cp docker/.env.example docker/.env
# Edit docker/.env:
#   APP_ENV=production
#   QDRANT_URL=http://qdrant:6333                           (Docker internal hostname)
#   API_KEYS=["your-secret-key-for-doctors"]
#   CORS_ORIGINS=["https://your-space.hf.space"]           (your HF Spaces URL)
#   OPENAI_API_KEY=...
#   COHERE_API_KEY=...
```

### 5 — Start the stack

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env up -d
```

### 6 — Set up HTTPS with Caddy

```bash
# Install Caddy
sudo apt install caddy

# /etc/caddy/Caddyfile
your-domain.com {
    reverse_proxy localhost:8000    # API
}
ui.your-domain.com {
    reverse_proxy localhost:8501    # UI (if not using HF Spaces)
}
```

---

## HuggingFace Spaces (UI only)

Deploy only `rag_ui.py` on HF Spaces (free tier available).
The API stays on the VPS. Set these secrets in the HF Spaces settings:

| Secret | Value |
|---|---|
| `API_BACKEND_URL` | `https://your-domain.com` |
| `API_KEY` | One of the keys from `API_KEYS` in your `.env` |

HF Spaces `requirements.txt` (minimal — only what the UI needs):
```
streamlit>=1.28.0
requests>=2.31.0
plotly>=5.17.0
pandas>=2.1.0
numpy>=1.24.0
```

---

## Weekly Incremental Ingestion (automatic)

Already wired in `docker-compose.yml` via the `ingestion-cron` profile.
Runs every Sunday at 02:00 UTC, fetches papers from the past 7 days.

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env --profile ingestion-cron up -d
```

---

## Production Checklist

- [ ] `APP_ENV=production` in `.env` (disables `/docs`, enforces auth)
- [ ] `API_KEYS` set to non-empty list (doctors' access keys)
- [ ] `ADMIN_API_KEY` set (for `/eval/run` access)
- [ ] Qdrant not exposed on public port (firewall blocks 6333)
- [ ] PostgreSQL not exposed on public port (firewall blocks 5432)
- [ ] HTTPS enabled via Caddy or Nginx
- [ ] Qdrant snapshot backup scheduled (weekly)
- [ ] Langfuse tracing enabled (optional — add `--profile langfuse`)

---

## Backup Strategy

```bash
# Qdrant snapshot (run weekly on server)
curl -X POST http://localhost:6333/collections/urological_oncology_papers/snapshots

# PostgreSQL dump
docker exec uro-rag-postgres pg_dump -U rag rag > backup_$(date +%Y%m%d).sql
```
