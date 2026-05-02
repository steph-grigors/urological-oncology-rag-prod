"""
Ingestion pipeline orchestrator.

Wires together fetch → parse → chunk → extract_metadata → embed into a
single callable that can be run from the CLI, a cron job, or triggered via
the `/eval/run` API endpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.embed import EmbedSummary, embed_chunks, ensure_collection
from src.ingestion.fetch import MESH_TERMS, fetch_batch, search_pmc

logger = logging.getLogger(__name__)

SUPPORTED_TOPICS = list(MESH_TERMS.keys())
_DEFAULT_CHECKPOINT = "data/ingestion_state.json"


@dataclass
class TopicSummary:
    papers_fetched: int = 0
    papers_skipped: int = 0
    chunks_produced: int = 0
    chunks_embedded: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class IngestionSummary:
    topics: dict[str, TopicSummary] = field(default_factory=dict)
    total_papers: int = 0
    total_chunks: int = 0
    total_embedded: int = 0
    elapsed_seconds: float = 0.0
    estimated_cost_usd: float = 0.0


def run_ingestion(
    topics: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
    max_papers_per_topic: int = 300,
    skip_metadata_extraction: bool = False,
    checkpoint_path: str = _DEFAULT_CHECKPOINT,
    dry_run: bool = False,
    openai_client=None,
    qdrant_client=None,
    collection: str = "urological_oncology_papers",
    ncbi_api_key: str = "",
) -> IngestionSummary:
    """Run the full pipeline. Returns aggregate stats."""
    t0 = time.monotonic()
    topics = topics or SUPPORTED_TOPICS
    summary = IngestionSummary()

    checkpoint = _load_checkpoint(checkpoint_path)
    ingested_ids: set[str] = set(checkpoint.get("ingested_ids", []))

    if qdrant_client and not dry_run:
        ensure_collection(qdrant_client, collection)

    meta_extractor = None
    if not skip_metadata_extraction and openai_client:
        try:
            from src.ingestion.extract_metadata import MetadataExtractor
            meta_extractor = MetadataExtractor(openai_client=openai_client)
        except Exception as exc:
            logger.warning("MetadataExtractor unavailable: %s", exc)

    for topic in topics:
        topic_sum = TopicSummary()
        summary.topics[topic] = topic_sum

        mesh_query = MESH_TERMS.get(topic, f'"{topic} cancer"[Title/Abstract]')
        logger.info("Searching PMC: topic=%r max=%d", topic, max_papers_per_topic)

        pmc_ids = search_pmc(
            mesh_query,
            max_results=max_papers_per_topic,
            date_range=date_range,
            ncbi_api_key=ncbi_api_key,
        )

        new_ids = [pid for pid in pmc_ids if pid not in ingested_ids]
        topic_sum.papers_skipped = len(pmc_ids) - len(new_ids)
        logger.info(
            "topic=%r: %d new / %d already ingested", topic, len(new_ids), topic_sum.papers_skipped
        )

        if dry_run:
            topic_sum.papers_fetched = len(new_ids)
            logger.info("[dry-run] Would process %d papers for topic=%r", len(new_ids), topic)
            summary.total_papers += len(new_ids)
            continue

        all_topic_chunks: list = []

        for pmc_id, xml in fetch_batch(new_ids, ncbi_api_key=ncbi_api_key):
            if xml is None:
                logger.warning("No XML for %s — skipping", pmc_id)
                continue

            try:
                from src.ingestion.parse import parse_paper
                paper = parse_paper(xml)
            except Exception as exc:
                logger.warning("parse_paper failed for %s: %s", pmc_id, exc)
                continue

            cancer_type = [topic]
            study_design = "unknown"
            sample_size: int | None = None
            primary_outcome: str | None = None

            if meta_extractor:
                try:
                    meta = meta_extractor.extract(paper)
                    cancer_type = getattr(meta, "cancer_types", None) or [topic]
                    study_design = getattr(meta, "study_design", None) or "unknown"
                    sample_size = getattr(meta, "sample_size", None)
                    primary_outcome = getattr(meta, "primary_outcome", None)
                except Exception as exc:
                    logger.warning("MetadataExtractor failed for %s: %s", pmc_id, exc)

            from src.ingestion.chunk import chunk_paper
            chunks = chunk_paper(
                paper,
                cancer_type=cancer_type,
                study_design=study_design,
                sample_size=sample_size,
                primary_outcome=primary_outcome,
            )

            all_topic_chunks.extend(chunks)
            topic_sum.papers_fetched += 1
            topic_sum.chunks_produced += len(chunks)
            ingested_ids.add(pmc_id)

            checkpoint["ingested_ids"] = list(ingested_ids)
            _save_checkpoint(checkpoint_path, checkpoint)

        if all_topic_chunks and openai_client and qdrant_client:
            embed_sum: EmbedSummary = embed_chunks(
                all_topic_chunks, openai_client, qdrant_client, collection
            )
            topic_sum.chunks_embedded = embed_sum.embedded
            topic_sum.estimated_cost_usd = embed_sum.estimated_cost_usd

        summary.total_papers += topic_sum.papers_fetched
        summary.total_chunks += topic_sum.chunks_produced
        summary.total_embedded += topic_sum.chunks_embedded
        summary.estimated_cost_usd += topic_sum.estimated_cost_usd

    summary.elapsed_seconds = time.monotonic() - t0
    logger.info(
        "Ingestion complete: papers=%d chunks=%d embedded=%d cost=$%.4f elapsed=%.1fs",
        summary.total_papers, summary.total_chunks, summary.total_embedded,
        summary.estimated_cost_usd, summary.elapsed_seconds,
    )
    return summary


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint(path: str) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"ingested_ids": []}


def _save_checkpoint(path: str, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.ingestion.pipeline",
        description="Urological Oncology RAG — ingestion pipeline",
    )
    p.add_argument("--mode", choices=["full", "incremental"], default="incremental")
    p.add_argument("--cancer-types", nargs="+", choices=SUPPORTED_TOPICS, dest="cancer_types")
    p.add_argument("--since-date", help="Start date YYYY/MM/DD (incremental mode)", dest="since_date")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT)
    p.add_argument("--skip-metadata", action="store_true", dest="skip_metadata")
    p.add_argument("--collection", default="urological_oncology_papers")
    return p


def _main() -> None:
    from src.observability.logging import setup_logging
    setup_logging()

    args = _build_parser().parse_args()

    date_range = None
    if args.mode == "incremental" and args.since_date:
        date_range = (args.since_date, "3000/01/01")

    openai_client = None
    qdrant_client = None

    try:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            openai_client = openai.OpenAI(api_key=api_key)
    except ImportError:
        logger.warning("openai not installed — embeddings disabled")

    try:
        from qdrant_client import QdrantClient
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        qdrant_client = QdrantClient(url=qdrant_url)
    except ImportError:
        logger.warning("qdrant_client not installed — upsert disabled")

    summary = run_ingestion(
        topics=args.cancer_types,
        date_range=date_range,
        max_papers_per_topic=args.limit,
        skip_metadata_extraction=args.skip_metadata,
        checkpoint_path=args.checkpoint,
        dry_run=args.dry_run,
        openai_client=openai_client,
        qdrant_client=qdrant_client,
        collection=args.collection,
        ncbi_api_key=os.environ.get("NCBI_API_KEY", ""),
    )

    print("\n=== Ingestion Summary ===")
    print(f"Mode: {args.mode}{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"Papers:   {summary.total_papers}")
    print(f"Chunks:   {summary.total_chunks}")
    print(f"Embedded: {summary.total_embedded}")
    print(f"Cost:     ${summary.estimated_cost_usd:.4f}")
    print(f"Elapsed:  {summary.elapsed_seconds:.1f}s\n")
    for topic, ts in summary.topics.items():
        print(
            f"  {topic}: fetched={ts.papers_fetched} "
            f"skipped={ts.papers_skipped} chunks={ts.chunks_produced}"
        )


if __name__ == "__main__":
    _main()
