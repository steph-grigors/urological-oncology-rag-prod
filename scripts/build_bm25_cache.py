"""
One-shot script to build and save the BM25 cache to data/bm25_cache.pkl.

Run this ONCE from a terminal outside VS Code to avoid IDE crashes:

    python scripts/build_bm25_cache.py

Qdrant must be running (docker compose up qdrant).
The cache is saved to data/bm25_cache.pkl, which is mounted into the API
container at /app/data/bm25_cache.pkl.  After this runs, the API loads in
seconds instead of scrolling 685K chunks at startup.
"""

import sys
import time
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient

from src.db.vector_store import QdrantStore
from src.retrieval.bm25_search import BM25Search

QDRANT_URL = "http://localhost:6333"
COLLECTION = "urological_oncology_papers"
CACHE_PATH = "data/bm25_cache.pkl"

if __name__ == "__main__":
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)
    store = QdrantStore(client, collection_name=COLLECTION)

    count = store.count()
    print(f"Collection has {count:,} points.")

    cache = Path(CACHE_PATH)
    if cache.exists():
        print(f"Cache already exists at {CACHE_PATH} — delete it first to rebuild.")
        sys.exit(0)

    print("Scrolling Qdrant and building BM25 index (this takes ~90 min on 685K chunks)...")
    t0 = time.time()
    bm25 = BM25Search.from_qdrant(store, cache_path=CACHE_PATH)
    elapsed = time.time() - t0

    print(f"Done in {elapsed/60:.1f} min. Cache saved to {CACHE_PATH} ({cache.stat().st_size / 1e6:.0f} MB).")
    print("Restart the API container to pick it up:")
    print("  docker compose -f docker/docker-compose.yml --env-file docker/.env restart api")
