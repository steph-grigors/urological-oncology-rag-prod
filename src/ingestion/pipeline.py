"""
Ingestion pipeline orchestrator.

Wires together fetch → parse → chunk → extract_metadata → embed into a
single callable that can be run from the CLI, a cron job, or triggered via
the `/eval/run` API endpoint.

Pipeline steps:
    1. fetch.search_pmc()         — collect PMC IDs for each topic
    2. fetch.fetch_batch()        — download XML in parallel
    3. parse.parse_paper()        — structured ParsedPaper per article
    4. chunk.chunk_paper()        — produce Chunk list per paper
    5. extract_metadata()         — LLM enrichment (optional, toggled by flag)
    6. embed.embed_chunks()       — embed + upsert to Qdrant

Checkpointing:
    Progress is checkpointed to a JSON file (default: data/ingestion_state.json)
    after each paper so the pipeline can resume after interruption without
    re-fetching or re-embedding already-processed papers.

Public API (to be implemented):
    run_ingestion(
        topics: list[str] | None = None,   # None → all SUPPORTED_TOPICS
        date_range: tuple[str, str] | None = None,
        max_papers_per_topic: int = 300,
        skip_metadata_extraction: bool = False,
        checkpoint_path: str = "data/ingestion_state.json",
    ) -> IngestionSummary
        Run the full pipeline. Returns aggregate stats.

    IngestionSummary(dataclass)
        topics: dict[str, TopicSummary]
        total_papers: int
        total_chunks: int
        total_embedded: int
        elapsed_seconds: float
        estimated_cost_usd: float
"""
