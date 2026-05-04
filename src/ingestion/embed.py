"""
Embedding generation and Qdrant upsert module.

Replaces `data_embeddings_scaled.py` with Qdrant as the vector store.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 1536          # text-embedding-3-small
_COST_PER_1K_TOKENS = 0.00002  # $0.02/1M tokens
_TOKENS_PER_WORD = 1.3         # rough estimate for cost tracking

try:
    from qdrant_client import QdrantClient  # noqa: F401
    from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: F401
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False


@dataclass
class EmbedSummary:
    total_chunks: int = 0
    embedded: int = 0
    skipped: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0
    estimated_cost_usd: float = 0.0


def ensure_collection(qdrant_client, collection: str, dim: int = _EMBEDDING_DIM) -> None:
    """Create the Qdrant collection with HNSW config if it doesn't already exist."""
    if not _QDRANT_AVAILABLE:
        raise ImportError("qdrant_client is not installed")

    from qdrant_client.models import Distance, VectorParams

    existing = {c.name for c in qdrant_client.get_collections().collections}
    if collection in existing:
        logger.info("ensure_collection: %r already exists", collection)
        return

    qdrant_client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    logger.info("ensure_collection: created %r dim=%d", collection, dim)


def embed_chunks(
    chunks: list,
    openai_client,
    qdrant_client,
    collection: str,
    batch_size: int = 100,
    qdrant_batch_size: int = 500,
    model: str = "text-embedding-3-small",
) -> EmbedSummary:
    """Generate embeddings for all chunks and upsert into Qdrant."""
    if not _QDRANT_AVAILABLE:
        raise ImportError("qdrant_client is not installed")

    from qdrant_client.models import PointStruct

    summary = EmbedSummary(total_chunks=len(chunks))
    t0 = time.monotonic()
    points_buffer: list = []

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start: batch_start + batch_size]
        texts = [_chunk_input_text(c) for c in batch]

        embeddings = _embed_with_retry(openai_client, texts, model)
        if embeddings is None:
            summary.failed += len(batch)
            logger.error("embed_chunks: batch failed at start=%d", batch_start)
            continue

        for chunk, vec in zip(batch, embeddings):
            points_buffer.append(
                PointStruct(
                    id=_stable_uuid(chunk.id),
                    vector=vec,
                    payload={
                        **_metadata_dict(chunk.metadata),
                        "chunk_id": chunk.id,
                        "text": chunk.text,
                    },
                )
            )
            summary.embedded += 1

        if len(points_buffer) >= qdrant_batch_size:
            _upsert_with_retry(qdrant_client, collection, points_buffer)
            points_buffer = []

        total_words = sum(len(t.split()) for t in texts)
        summary.estimated_cost_usd += (
            total_words * _TOKENS_PER_WORD / 1000
        ) * _COST_PER_1K_TOKENS

    if points_buffer:
        _upsert_with_retry(qdrant_client, collection, points_buffer)

    summary.elapsed_seconds = time.monotonic() - t0
    logger.info(
        "embed_chunks: total=%d embedded=%d failed=%d cost=$%.4f elapsed=%.1fs",
        summary.total_chunks, summary.embedded, summary.failed,
        summary.estimated_cost_usd, summary.elapsed_seconds,
    )
    return summary


# ── Private helpers ───────────────────────────────────────────────────────────

def _chunk_input_text(chunk) -> str:
    """Build the text to embed: context_prefix (if set) + chunk text."""
    prefix = getattr(chunk, "context_prefix", "")
    if prefix:
        return f"{prefix}\n{chunk.text}"
    meta = getattr(chunk, "metadata", None)
    title = getattr(meta, "title", "") or ""
    section = getattr(meta, "section", "") or ""
    if title or section:
        return f"{title} | {section}\n{chunk.text}"
    return chunk.text


def _metadata_dict(meta) -> dict:
    if hasattr(meta, "__dataclass_fields__"):
        result: dict = {}
        for key in meta.__dataclass_fields__:
            val = getattr(meta, key)
            if val is not None:
                result[key] = val
        return result
    if hasattr(meta, "__dict__"):
        return {k: v for k, v in meta.__dict__.items() if v is not None}
    return {}


def _stable_uuid(chunk_id: str) -> str:
    """Return a deterministic UUID string derived from chunk_id."""
    digest = hashlib.md5(chunk_id.encode()).digest()
    return str(uuid.UUID(bytes=digest))


def _embed_with_retry(openai_client, texts: list[str], model: str, max_retries: int = 4) -> list | None:
    for attempt in range(max_retries):
        try:
            response = openai_client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]
        except Exception as exc:
            wait = 2.0 ** attempt
            logger.warning(
                "embed retry %d/%d error=%s, sleeping %.1fs",
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)
    return None


def _upsert_with_retry(qdrant_client, collection: str, points: list, max_retries: int = 3) -> None:
    for attempt in range(max_retries):
        try:
            qdrant_client.upsert(collection_name=collection, points=points)
            return
        except Exception as exc:
            wait = 2.0 ** attempt
            logger.warning(
                "upsert retry %d/%d error=%s, sleeping %.1fs",
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)
    logger.error("upsert failed after %d attempts (%d points)", max_retries, len(points))
