"""
Embedding generation and Qdrant upsert module.

Replaces `data_embeddings_scaled.py` with Qdrant as the vector store.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.db.point_id import chunk_point_id

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
    embedded: int = 0          # points confirmed written to Qdrant
    skipped: int = 0
    failed: int = 0            # points that never reached Qdrant
    upsert_failures: int = 0   # number of batches whose upsert gave up
    elapsed_seconds: float = 0.0
    estimated_cost_usd: float = 0.0

    @property
    def all_persisted(self) -> bool:
        """True when every embedded chunk is known to be in Qdrant.

        The pipeline checkpoints a batch of papers as ingested only when this
        holds; otherwise the papers are left un-checkpointed so the next run
        picks them up again.
        """
        return self.upsert_failures == 0 and self.failed == 0


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
            if not _upsert_with_retry(qdrant_client, collection, points_buffer):
                summary.embedded -= len(points_buffer)
                summary.failed += len(points_buffer)
                summary.upsert_failures += 1
            points_buffer = []

        total_words = sum(len(t.split()) for t in texts)
        summary.estimated_cost_usd += (
            total_words * _TOKENS_PER_WORD / 1000
        ) * _COST_PER_1K_TOKENS

    if points_buffer:
        if not _upsert_with_retry(qdrant_client, collection, points_buffer):
            summary.embedded -= len(points_buffer)
            summary.failed += len(points_buffer)
            summary.upsert_failures += 1

    summary.elapsed_seconds = time.monotonic() - t0
    log = logger.info if summary.all_persisted else logger.error
    log(
        "embed_chunks: total=%d embedded=%d failed=%d upsert_failures=%d "
        "cost=$%.4f elapsed=%.1fs",
        summary.total_chunks, summary.embedded, summary.failed,
        summary.upsert_failures, summary.estimated_cost_usd,
        summary.elapsed_seconds,
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
    """Return a deterministic UUID string derived from chunk_id.

    Delegates to src/db/point_id.chunk_point_id so this path and the db layer
    cannot drift apart again. The derivation is unchanged, so every point
    already in production keeps its id.
    """
    return chunk_point_id(chunk_id)


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


def _upsert_with_retry(qdrant_client, collection: str, points: list, max_retries: int = 3) -> bool:
    """Upsert `points`, returning True on success and False once retries are
    exhausted.

    This used to return None either way, so a permanent failure logged an
    error and the caller carried on as though the points had landed. The
    chunks had already been counted as embedded, the pipeline then checkpointed
    the papers as ingested, and the next run skipped them -- the chunks were
    gone with nothing but a log line to say so.
    """
    for attempt in range(max_retries):
        try:
            qdrant_client.upsert(collection_name=collection, points=points)
            return True
        except Exception as exc:
            wait = 2.0 ** attempt
            logger.warning(
                "upsert retry %d/%d error=%s, sleeping %.1fs",
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)
    logger.error("upsert failed after %d attempts (%d points)", max_retries, len(points))
    return False
