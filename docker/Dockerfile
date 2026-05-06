# ── Stage 1: builder — install all Python dependencies ───────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime — lean image with only what's needed ─────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime-only native libs: curl (healthcheck) + libpq5 (asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application code (no chroma_db or legacy data)
COPY config/ ./config/
COPY src/ ./src/
COPY rag_ui.py ./rag_ui.py

# Persistent data directory (evaluation outputs, metadata cache, ingestion state)
RUN mkdir -p /app/data/evaluation

# Non-root user — never run as root in production
RUN addgroup --system app && adduser --system --ingroup app app \
    && chown -R app:app /app
USER app

# Liveness probe via the /health/live endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

EXPOSE 8000

# Single worker locally (WSL2 RAM constraint); scale to 2-4 on the cloud VPS
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
