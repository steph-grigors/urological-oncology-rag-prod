"""
Embedding generation and Qdrant upsert module.

Replaces `data_embeddings_scaled.py` with Qdrant as the vector store and
adds support for the `context_prefix` embedding strategy from `chunk.py`.

Key improvements over the existing implementation:
- Targets Qdrant instead of ChromaDB for production scalability.
- Embeds `chunk.context_prefix + "\\n" + chunk.text` rather than bare text
  so the embedding captures title + section signals.
- Batches OpenAI API calls (batch size 100) with exponential-backoff retry.
- Upserts in configurable batch sizes to Qdrant (default 500 points/batch).
- Idempotent: uses the deterministic chunk `id` as the Qdrant point ID so
  re-runs only overwrite changed chunks.
- Tracks and reports cost estimate based on token counts.

Public API (to be implemented):
    embed_chunks(
        chunks: list[Chunk],
        openai_client: OpenAI,
        qdrant_client: QdrantClient,
        collection: str,
        batch_size: int = 100,
    ) -> EmbedSummary
        Generate embeddings for all chunks and upsert into Qdrant.
        Returns a summary with counts, elapsed time, and cost estimate.

    ensure_collection(qdrant_client: QdrantClient, collection: str, dim: int) -> None
        Create the Qdrant collection with HNSW config if it doesn't exist.
        Called once at pipeline startup; safe to call repeatedly.

    EmbedSummary(dataclass)
        total_chunks: int
        embedded: int
        skipped: int          # already up-to-date (idempotent re-run)
        failed: int
        elapsed_seconds: float
        estimated_cost_usd: float
"""
