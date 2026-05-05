"""
Ingestion pipeline orchestrator.

Wires together fetch → parse → chunk → extract_metadata → embed into a
single callable that can be run from the CLI, a cron job, or triggered via
the `/eval/run` API endpoint.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from src.ingestion.embed import EmbedSummary, embed_chunks, ensure_collection
from src.ingestion.fetch import MESH_TERMS, fetch_batch, search_pmc

logger = logging.getLogger(__name__)

SUPPORTED_TOPICS = list(MESH_TERMS.keys())
_DEFAULT_CHECKPOINT = "data/ingestion_state.json"
_PROGRESS_PATH = "data/ingestion_progress.json"

# Rolling batch: embed + checkpoint after every N papers rather than at end of
# topic. Keeps memory bounded and ensures the checkpoint always reflects what
# is actually in Qdrant (avoids silent data loss on crash-resume).
_EMBED_BATCH_SIZE = 50
_META_WORKERS = 2  # parallel threads for GPT-4o-mini metadata extraction


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
    skip_low_quality: bool = True,
    checkpoint_path: str = _DEFAULT_CHECKPOINT,
    rejected_path: str = "data/ingestion_rejected.json",
    dry_run: bool = False,
    openai_client=None,
    qdrant_client=None,
    collection: str = "urological_oncology_papers",
    ncbi_api_key: str = "",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestionSummary:
    """Run the full pipeline. Returns aggregate stats."""
    from config.settings import Settings
    _settings = Settings()
    _chunk_size = chunk_size or _settings.chunk_size_words
    _chunk_overlap = chunk_overlap or _settings.chunk_overlap_words

    t0 = time.monotonic()
    topics = topics or SUPPORTED_TOPICS
    summary = IngestionSummary()

    checkpoint = _load_checkpoint(checkpoint_path)
    ingested_ids: set[str] = set(checkpoint.get("ingested_ids", []))
    rejected: list[dict] = _load_rejected(rejected_path)

    if qdrant_client and not dry_run:
        ensure_collection(qdrant_client, collection)

    meta_extractor = None
    if not skip_metadata_extraction and openai_client:
        try:
            from src.ingestion.extract_metadata import MetadataExtractor
            meta_extractor = MetadataExtractor(openai_client=openai_client)
        except Exception as exc:
            logger.warning("MetadataExtractor unavailable: %s", exc)

    # ── Progress tracking state ───────────────────────────────────────────────
    progress: dict = {
        "status": "running",
        "started_at": datetime.datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "current_topic": None,
        "topics": {
            t: {"status": "pending", "papers_total": 0, "papers_fetched": 0,
                "papers_skipped": 0, "chunks_produced": 0, "chunks_embedded": 0,
                "estimated_cost_usd": 0.0}
            for t in topics
        },
        "totals": {
            "papers_fetched": 0, "papers_total": 0,
            "chunks_produced": 0, "chunks_embedded": 0,
            "estimated_cost_usd": 0.0, "elapsed_seconds": 0.0,
        },
    }
    _write_progress(_PROGRESS_PATH, progress)

    topic_bar = tqdm(topics, desc="Ingestion", unit="topic", position=0, leave=True)

    for topic in topic_bar:
        topic_bar.set_description(f"Topics [{topic}]")
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

        progress["current_topic"] = topic
        progress["topics"][topic].update({
            "status": "running",
            "papers_total": len(new_ids),
            "papers_skipped": topic_sum.papers_skipped,
        })
        progress["totals"]["papers_total"] += len(new_ids)
        _write_progress(_PROGRESS_PATH, progress)

        if dry_run:
            topic_sum.papers_fetched = len(new_ids)
            logger.info("[dry-run] Would process %d papers for topic=%r", len(new_ids), topic)
            summary.total_papers += len(new_ids)
            progress["topics"][topic]["status"] = "complete"
            _write_progress(_PROGRESS_PATH, progress)
            continue

        # Collect parsed+quality-passed papers into rolling batches.
        # Each batch is: metadata (parallel) → chunk → embed → checkpoint.
        # The checkpoint is written AFTER a successful Qdrant upsert so that a
        # crash-resume never leaves papers in the checkpoint but missing from
        # the vector DB.
        pending: list[tuple[str, object]] = []  # (pmc_id, ParsedPaper)

        from src.ingestion.parse import parse_paper
        from src.ingestion.quality import score_paper_quality
        from src.ingestion.chunk import chunk_paper

        paper_bar = tqdm(
            total=len(new_ids),
            desc=f"  {topic}",
            unit="paper",
            position=1,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
        )

        def _flush(batch: list[tuple[str, object]]) -> None:
            if not batch:
                return

            # ── Parallel metadata extraction ──────────────────────────────
            meta_map: dict[str, object] = {}
            if meta_extractor:
                with ThreadPoolExecutor(max_workers=_META_WORKERS) as pool:
                    futures = {
                        pool.submit(meta_extractor.extract, paper): pmc_id
                        for pmc_id, paper in batch
                    }
                    for fut in as_completed(futures):
                        pid = futures[fut]
                        try:
                            meta_map[pid] = fut.result()
                        except Exception as exc:
                            logger.warning("Metadata extraction failed for %s: %s", pid, exc)

            # ── Chunk all papers in batch ─────────────────────────────────
            batch_chunks: list = []
            for pmc_id, paper in batch:
                meta = meta_map.get(pmc_id)
                chunks = chunk_paper(
                    paper,
                    cancer_type=getattr(meta, "cancer_types", None) or [topic],
                    study_design=getattr(meta, "study_design", None) or "unknown",
                    sample_size=getattr(meta, "sample_size", None),
                    primary_outcome=getattr(meta, "primary_outcome", None),
                    intervention=getattr(meta, "intervention", None),
                    comparator=getattr(meta, "comparator", None),
                    chunk_size=_chunk_size,
                    overlap=_chunk_overlap,
                )
                batch_chunks.extend(chunks)
                topic_sum.papers_fetched += 1
                topic_sum.chunks_produced += len(chunks)

            # ── Embed + upsert, then checkpoint ───────────────────────────
            # Checkpoint is written ONLY after a successful Qdrant upsert.
            # If no embedding clients are available the batch is skipped
            # entirely so the papers remain un-checkpointed and will be
            # re-processed (and embedded) on the next run.
            if not openai_client or not qdrant_client:
                logger.warning(
                    "Embedding clients unavailable — batch of %d papers NOT checkpointed. "
                    "Re-run with OPENAI_API_KEY and QDRANT_URL set to embed them.",
                    len(batch),
                )
                return

            if batch_chunks:
                embed_sum: EmbedSummary = embed_chunks(
                    batch_chunks, openai_client, qdrant_client, collection
                )
                topic_sum.chunks_embedded += embed_sum.embedded
                topic_sum.estimated_cost_usd += embed_sum.estimated_cost_usd

            for pmc_id, _ in batch:
                ingested_ids.add(pmc_id)
            checkpoint["ingested_ids"] = list(ingested_ids)
            _save_checkpoint(checkpoint_path, checkpoint)
            logger.info(
                "Flushed batch: topic=%r papers=%d chunks=%d",
                topic, len(batch), len(batch_chunks),
            )

            # ── Update progress file after every successful flush ─────────
            elapsed = time.monotonic() - t0
            progress["topics"][topic].update({
                "papers_fetched": topic_sum.papers_fetched,
                "chunks_produced": topic_sum.chunks_produced,
                "chunks_embedded": topic_sum.chunks_embedded,
                "estimated_cost_usd": topic_sum.estimated_cost_usd,
            })
            progress["totals"].update({
                "papers_fetched": sum(v["papers_fetched"] for v in progress["topics"].values()),
                "chunks_produced": sum(v["chunks_produced"] for v in progress["topics"].values()),
                "chunks_embedded": sum(v["chunks_embedded"] for v in progress["topics"].values()),
                "estimated_cost_usd": sum(v["estimated_cost_usd"] for v in progress["topics"].values()),
                "elapsed_seconds": round(elapsed, 1),
            })
            progress["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            _write_progress(_PROGRESS_PATH, progress)

        for pmc_id, xml in fetch_batch(new_ids, ncbi_api_key=ncbi_api_key):
            paper_bar.update(1)

            if xml is None:
                logger.warning("No XML for %s — skipping", pmc_id)
                continue

            try:
                paper = parse_paper(xml)
            except Exception as exc:
                logger.warning("parse_paper failed for %s: %s", pmc_id, exc)
                continue

            if paper is None:
                logger.warning("parse_paper returned None for %s — skipping", pmc_id)
                continue

            if skip_low_quality:
                quality = score_paper_quality(paper)
                if not quality.passed:
                    logger.info(
                        "Quality gate rejected %s (score=%.2f): %s",
                        pmc_id, quality.score, "; ".join(quality.reasons),
                    )
                    rejected.append({
                        "pmc_id": pmc_id,
                        "topic": topic,
                        "score": quality.score,
                        "reasons": quality.reasons,
                    })
                    _save_rejected(rejected_path, rejected)
                    continue

            pending.append((pmc_id, paper))
            paper_bar.set_postfix(
                chunks=topic_sum.chunks_produced,
                embedded=topic_sum.chunks_embedded,
                cost=f"${topic_sum.estimated_cost_usd:.3f}",
            )

            if len(pending) >= _EMBED_BATCH_SIZE:
                _flush(pending)
                pending.clear()

        _flush(pending)  # final partial batch
        paper_bar.close()

        progress["topics"][topic]["status"] = "complete"
        _write_progress(_PROGRESS_PATH, progress)

        summary.total_papers += topic_sum.papers_fetched
        summary.total_chunks += topic_sum.chunks_produced
        summary.total_embedded += topic_sum.chunks_embedded
        summary.estimated_cost_usd += topic_sum.estimated_cost_usd

    topic_bar.close()

    summary.elapsed_seconds = time.monotonic() - t0
    progress["status"] = "complete"
    progress["totals"]["elapsed_seconds"] = round(summary.elapsed_seconds, 1)
    progress["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    _write_progress(_PROGRESS_PATH, progress)

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


def _load_rejected(path: str) -> list:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return []


def _save_rejected(path: str, records: list) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(records, indent=2))


def _write_progress(path: str, data: dict) -> None:
    """Write progress snapshot; silently ignores I/O errors to never block the pipeline."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(p)
    except Exception as exc:
        logger.debug("Could not write progress file: %s", exc)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.ingestion.pipeline",
        description="Urological Oncology RAG — ingestion pipeline",
    )
    p.add_argument("--mode", choices=["full", "incremental"], default="incremental")
    p.add_argument("--cancer-types", nargs="+", choices=SUPPORTED_TOPICS, dest="cancer_types")
    p.add_argument("--since-date", help="Only ingest papers published on or after YYYY/MM/DD", dest="since_date")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT)
    p.add_argument("--skip-metadata", action="store_true", dest="skip_metadata")
    p.add_argument("--collection", default="urological_oncology_papers")
    return p


def _main() -> None:
    from src.observability.logging import setup_logging
    setup_logging()

    # Load docker/.env if present so the pipeline can be run directly from the
    # shell without manually exporting variables.
    _env_file = Path(__file__).resolve().parents[2] / "docker" / ".env"
    if _env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env_file, override=False)  # shell exports take precedence
            logger.info("Loaded env from %s", _env_file)
        except ImportError:
            pass  # python-dotenv not installed — fall back to shell env

    args = _build_parser().parse_args()

    date_range = None
    if args.since_date:
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
