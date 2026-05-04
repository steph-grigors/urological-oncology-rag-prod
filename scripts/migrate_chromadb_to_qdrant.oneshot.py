#!/usr/bin/env python3
"""
One-time migration: ChromaDB → Qdrant.

ChromaDB is not installed in the current environment, so this script reads
text and metadata directly from the SQLite backing store at
`chroma_db_scaled/chroma.sqlite3`, re-embeds using OpenAI, and upserts
to Qdrant.

Re-embedding is safe because:
 - Both the original ChromaDB collection and the new system use the same
   model (text-embedding-3-small, 1536 dimensions).
 - The cost is ~$0.11 for 41 970 chunks at the current token rate.

Usage:
    OPENAI_API_KEY=... QDRANT_URL=http://localhost:6333 \\
        python scripts/migrate_chromadb_to_qdrant.py \\
        [--sqlite-path chroma_db_scaled/chroma.sqlite3] \\
        [--collection urological_oncology_papers] \\
        [--batch-size 100] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIM = 1536
_COST_PER_1K_TOKENS = 0.00002
_TOKENS_PER_WORD = 1.3
_DEFAULT_SQLITE = "chroma_db_scaled/chroma.sqlite3"
_DEFAULT_COLLECTION = "urological_oncology_papers"


@dataclass
class MigrationReport:
    source_count: int = 0
    migrated: int = 0
    failed: int = 0
    skipped: int = 0
    estimated_cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


# ── ChromaDB SQLite reader ────────────────────────────────────────────────────

def read_chromadb_records(sqlite_path: str) -> list[dict]:
    """Read all records from ChromaDB's SQLite store without the chromadb package."""
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    # Build a map: embedding_id → {key: value}
    logger.info("Reading embedding_metadata from %s", sqlite_path)
    meta_by_id: dict[str, dict] = {}

    rows = conn.execute(
        "SELECT e.embedding_id, em.key, em.string_value, em.int_value, em.float_value "
        "FROM embeddings e "
        "JOIN embedding_metadata em ON e.id = em.id "
        "ORDER BY e.embedding_id"
    ).fetchall()

    for row in rows:
        eid = row["embedding_id"]
        if eid not in meta_by_id:
            meta_by_id[eid] = {"embedding_id": eid}
        key = row["key"]
        # Pick whichever value is non-null
        val = row["string_value"] or (
            row["int_value"] if row["int_value"] is not None else row["float_value"]
        )
        meta_by_id[eid][key] = val

    conn.close()
    records = list(meta_by_id.values())
    logger.info("Loaded %d records from ChromaDB SQLite", len(records))
    return records


def _build_embed_text(record: dict) -> str:
    """Recreate the embedding input: 'title | section\ntext'."""
    title = record.get("title", "") or ""
    section = record.get("section_name", "") or ""
    doc = record.get("chroma:document", "") or ""
    if title or section:
        return f"{title} | {section}\n{doc}"
    return doc


def _record_to_payload(record: dict) -> dict:
    """Map ChromaDB metadata keys to the Qdrant payload schema."""
    pmc_id = record.get("pmc_id", "")
    return {
        "chunk_id": record["embedding_id"],
        "text": (record.get("chroma:document", "") or "")[:2000],
        "pmcid": pmc_id,
        "pmid": record.get("pmid", ""),
        "title": record.get("title", ""),
        "journal": record.get("journal", ""),
        "year": _int_or_none(record.get("year")),
        "section": record.get("section_name", ""),
        "topic": record.get("topic", ""),
        "cancer_type": [record["topic"]] if record.get("topic") else [],
        "chunk_index": _int_or_none(record.get("chunk_index")),
        "total_chunks": _int_or_none(record.get("total_chunks")),
        "doi": record.get("doi", ""),
        # Fields not available in original ChromaDB data — fill with defaults
        "study_design": "unknown",
        "evidence_level": 6,
        "chunk_type": "text",
    }


def _int_or_none(val) -> int | None:
    try:
        return int(val) if val is not None else None
    except (ValueError, TypeError):
        return None


def _stable_uuid(embedding_id: str) -> str:
    digest = hashlib.md5(embedding_id.encode()).digest()
    return str(uuid.UUID(bytes=digest))


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _embed_batch(openai_client, texts: list[str], max_retries: int = 4) -> list | None:
    for attempt in range(max_retries):
        try:
            resp = openai_client.embeddings.create(input=texts, model=_EMBEDDING_MODEL)
            return [item.embedding for item in resp.data]
        except Exception as exc:
            wait = 2.0 ** attempt
            logger.warning("embed retry %d/%d error=%s, sleep %.1fs", attempt + 1, max_retries, exc, wait)
            time.sleep(wait)
    return None


def _upsert_batch(qdrant_client, collection: str, points: list, max_retries: int = 3) -> bool:
    for attempt in range(max_retries):
        try:
            qdrant_client.upsert(collection_name=collection, points=points)
            return True
        except Exception as exc:
            wait = 2.0 ** attempt
            logger.warning("upsert retry %d/%d error=%s, sleep %.1fs", attempt + 1, max_retries, exc, wait)
            time.sleep(wait)
    return False


# ── Main migration logic ──────────────────────────────────────────────────────

def migrate(
    sqlite_path: str,
    openai_client,
    qdrant_client,
    collection: str,
    batch_size: int = 100,
    dry_run: bool = False,
) -> MigrationReport:
    report = MigrationReport()
    t0 = time.monotonic()

    if not Path(sqlite_path).exists():
        logger.error("SQLite path not found: %s", sqlite_path)
        report.errors.append(f"SQLite not found: {sqlite_path}")
        return report

    records = read_chromadb_records(sqlite_path)
    report.source_count = len(records)

    if dry_run:
        logger.info("[dry-run] Would migrate %d records — no writes performed", len(records))
        report.elapsed_seconds = time.monotonic() - t0
        return report

    # Ensure Qdrant collection exists
    try:
        from qdrant_client.models import Distance, VectorParams
        existing = {c.name for c in qdrant_client.get_collections().collections}
        if collection not in existing:
            qdrant_client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=_EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection %r", collection)
        else:
            logger.info("Collection %r already exists", collection)
    except Exception as exc:
        logger.error("Failed to ensure Qdrant collection: %s", exc)
        report.errors.append(str(exc))
        return report

    # Process in batches
    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start: batch_start + batch_size]
        texts = [_build_embed_text(r) for r in batch]

        embeddings = _embed_batch(openai_client, texts)
        if embeddings is None:
            report.failed += len(batch)
            msg = f"Embedding failed for batch starting at {batch_start}"
            report.errors.append(msg)
            logger.error(msg)
            continue

        from qdrant_client.models import PointStruct
        points = [
            PointStruct(
                id=_stable_uuid(r["embedding_id"]),
                vector=vec,
                payload=_record_to_payload(r),
            )
            for r, vec in zip(batch, embeddings)
        ]

        ok = _upsert_batch(qdrant_client, collection, points)
        if ok:
            report.migrated += len(batch)
        else:
            report.failed += len(batch)
            report.errors.append(f"Upsert failed for batch at {batch_start}")

        # Cost estimate
        total_words = sum(len(t.split()) for t in texts)
        report.estimated_cost_usd += (total_words * _TOKENS_PER_WORD / 1000) * _COST_PER_1K_TOKENS

        logger.info(
            "Progress: %d/%d (migrated=%d failed=%d)",
            min(batch_start + batch_size, len(records)), len(records),
            report.migrated, report.failed,
        )

    # Verify
    try:
        qdrant_count = qdrant_client.get_collection(collection).points_count
        logger.info(
            "Verification: ChromaDB=%d Qdrant=%d", report.source_count, qdrant_count
        )
        if qdrant_count < report.migrated:
            logger.warning(
                "Qdrant count (%d) < migrated (%d) — some points may be missing",
                qdrant_count, report.migrated,
            )
    except Exception as exc:
        logger.warning("Could not verify Qdrant count: %s", exc)

    report.elapsed_seconds = time.monotonic() - t0
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Migrate ChromaDB → Qdrant")
    p.add_argument("--sqlite-path", default=_DEFAULT_SQLITE)
    p.add_argument("--collection", default=_DEFAULT_COLLECTION)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key and not args.dry_run:
        logger.error("OPENAI_API_KEY not set — cannot embed. Use --dry-run to preview.")
        sys.exit(1)

    try:
        import openai
        openai_client = openai.OpenAI(api_key=openai_api_key)
    except ImportError:
        logger.error("openai package not installed")
        sys.exit(1)

    try:
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(url=args.qdrant_url)
    except ImportError:
        logger.error("qdrant_client package not installed")
        sys.exit(1)

    report = migrate(
        sqlite_path=args.sqlite_path,
        openai_client=openai_client,
        qdrant_client=qdrant_client,
        collection=args.collection,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    print("\n=== Migration Report ===")
    print(f"Source (ChromaDB): {report.source_count:,} chunks")
    print(f"Migrated:          {report.migrated:,}")
    print(f"Failed:            {report.failed:,}")
    print(f"Cost estimate:     ${report.estimated_cost_usd:.4f}")
    print(f"Elapsed:           {report.elapsed_seconds:.1f}s")
    if report.errors:
        print(f"\nErrors ({len(report.errors)}):")
        for e in report.errors[:10]:
            print(f"  - {e}")

    if report.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
