"""
PubMed Central (PMC) fetching module.

Replaces `data_collection_scaled.py` with a cleaner, async-capable interface.

Responsibilities:
- Search PMC via Entrez esearch for a given topic query and date range.
- Batch-fetch full-text XML using Entrez efetch with exponential-backoff
  retry logic.
- Respect NCBI rate limits (10 req/s with API key, 3 req/s without).
- Stream results so the caller can process papers incrementally without
  holding all XML in memory simultaneously.
- Yield raw XML strings; downstream parsing happens in `parse.py`.

Public API (to be implemented):
    search_pmc(query: str, max_results: int, date_range: tuple) -> list[str]
        Return a list of PMC IDs matching the query.

    fetch_fulltext(pmc_id: str) -> str | None
        Fetch the full-text XML for a single PMC article. Returns None on
        permanent failure (article not in OA subset, retracted, etc.).

    fetch_batch(pmc_ids: list[str], *, workers: int = 4) -> Iterator[tuple[str, str | None]]
        Yield (pmc_id, xml_str | None) pairs, fetching in parallel up to
        `workers` concurrent requests.
"""
