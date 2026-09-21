"""
Rebuild the BM25 disk cache from the live Qdrant collection.

Run this after an ingestion run. The cache is keyed on the collection's point
count, so adding chunks makes it stale; without this, the next API restart
re-scrolls the whole collection and rebuilds the index inline, which takes
several minutes of downtime instead of seconds.

Unlike scripts/build_bm25_cache.py, which is a one-shot that refuses to
overwrite an existing cache and assumes Qdrant is on localhost, this:

  - reads QDRANT_URL / QDRANT_COLLECTION from the environment, so it works
    inside the api container where Qdrant is at http://qdrant:6333
  - builds into a temporary directory and swaps it in, so there is never a
    window where the cache is missing or half-written
  - keeps the previous cache alongside, for a manual rollback

Usage (inside the api container):
    python scripts/rebuild_bm25_cache.py
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient

from src.db.vector_store import QdrantStore
from src.retrieval.bm25_search import BM25_CACHE_DIR, BM25Search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
logger = logging.getLogger("rebuild_bm25_cache")

_SCROLL_TIMEOUT_SECONDS = 600


def rebuild(cache_dir: str = BM25_CACHE_DIR) -> int:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    collection = os.environ.get("QDRANT_COLLECTION", "urological_oncology_papers")

    live = Path(cache_dir)
    new = live.with_name(live.name + "_new")
    previous = live.with_name(live.name + "_previous")

    t0 = time.monotonic()
    store = QdrantStore(
        QdrantClient(url=url, timeout=_SCROLL_TIMEOUT_SECONDS),
        collection_name=collection,
    )
    count = store.count()
    if count == 0:
        logger.error("Collection %r is empty — refusing to overwrite the cache", collection)
        return 1
    logger.info("Collection %r holds %d points", collection, count)

    chunks = store.scroll_all()
    logger.info("Scrolled %d chunks in %.0fs", len(chunks), time.monotonic() - t0)

    index = BM25Search(chunks)
    logger.info("Index built after %.0fs", time.monotonic() - t0)

    if new.exists():
        shutil.rmtree(new)
    index.save(str(new), count)
    if not (new / "meta.json").exists():
        logger.error("Save produced no meta.json — leaving the existing cache untouched")
        return 1

    if previous.exists():
        shutil.rmtree(previous)
    if live.exists():
        live.rename(previous)
    new.rename(live)
    logger.info(
        "Cache swapped in (%d chunks, %.0fs total). Previous kept at %s",
        count, time.monotonic() - t0, previous,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(rebuild())
