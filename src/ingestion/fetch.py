"""
PubMed Central (PMC) fetching module.

Replaces `data_collection_scaled.py` with a cleaner, async-capable interface.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_DEFAULT_RATE = 3    # req/sec without API key
_MAX_RETRIES = 4
_RETRY_BASE = 2.0    # seconds; actual wait = _RETRY_BASE ** attempt

# Publication-type filter: only high-evidence study designs relevant to clinical practice.
# Excludes case reports, editorials, letters, and other low-signal types.
_PT_FILTER = (
    '("Randomized Controlled Trial"[PT] OR "Meta-Analysis"[PT] OR '
    '"Systematic Review"[PT] OR "Practice Guideline"[PT] OR "Clinical Trial"[PT])'
)

# Retraction and language filters: safety-critical for a clinical decision-support tool.
_NOT_RETRACTED = 'NOT "Retracted Publication"[PT]'
_ENGLISH_ONLY  = 'AND "English"[Language]'

MESH_TERMS: dict[str, str] = {
    "prostate":   f'"Prostatic Neoplasms"[MeSH] AND {_PT_FILTER} {_NOT_RETRACTED} {_ENGLISH_ONLY}',
    "bladder":    f'"Urinary Bladder Neoplasms"[MeSH] AND {_PT_FILTER} {_NOT_RETRACTED} {_ENGLISH_ONLY}',
    "kidney":     f'"Kidney Neoplasms"[MeSH] AND {_PT_FILTER} {_NOT_RETRACTED} {_ENGLISH_ONLY}',
    "testicular": f'"Testicular Neoplasms"[MeSH] AND {_PT_FILTER} {_NOT_RETRACTED} {_ENGLISH_ONLY}',
    "penile":     f'"Penile Neoplasms"[MeSH] AND {_PT_FILTER} {_NOT_RETRACTED} {_ENGLISH_ONLY}',
    "adrenal":    f'"Adrenal Gland Neoplasms"[MeSH] AND {_PT_FILTER} {_NOT_RETRACTED} {_ENGLISH_ONLY}',
}


_PAGE_SIZE = 9999  # NCBI hard cap per esearch request


def search_pmc(
    query: str,
    max_results: int = 300,
    date_range: tuple[str, str] | None = None,
    ncbi_api_key: str = "",
) -> list[str]:
    """Return a list of PMC IDs matching the query.

    Uses NCBI's history server (usehistory=y + WebEnv pagination) to bypass
    the 10,000-result-per-request cap.  Results are fetched in pages of 9,999
    until max_results is reached or the server has no more IDs.
    """
    base_params: dict = {
        "db": "pmc",
        "term": query,
        "usehistory": "y",      # register result set server-side
        "retmax": 0,            # first call: just get count + WebEnv
        "retmode": "json",
    }
    if date_range:
        base_params["mindate"] = date_range[0]
        base_params["maxdate"] = date_range[1]
        base_params["datetype"] = "pdat"
    if ncbi_api_key:
        base_params["api_key"] = ncbi_api_key

    # ── Step 1: register query, get total count + WebEnv ─────────────────
    data = _get_json(f"{_BASE_URL}/esearch.fcgi", base_params)
    result = data.get("esearchresult", {})
    total = int(result.get("count", 0))
    web_env = result.get("webenv", "")
    query_key = result.get("querykey", "")

    if not web_env or not query_key:
        # Fallback: single-page fetch without history server
        ids = result.get("idlist", [])
        logger.info("search_pmc (no history) query=%r found %d IDs", query, len(ids))
        return [f"PMC{i}" for i in ids[:max_results]]

    to_fetch = min(total, max_results)
    logger.info(
        "search_pmc query=%r total_on_server=%d fetching=%d", query, total, to_fetch
    )

    # ── Step 2: paginate through results ─────────────────────────────────
    all_ids: list[str] = []
    retstart = 0

    while retstart < to_fetch:
        page_size = min(_PAGE_SIZE, to_fetch - retstart)
        page_params: dict = {
            "db": "pmc",
            "query_key": query_key,
            "WebEnv": web_env,
            "retstart": retstart,
            "retmax": page_size,
            "retmode": "json",
        }
        if ncbi_api_key:
            page_params["api_key"] = ncbi_api_key

        page_data = _get_json(f"{_BASE_URL}/esearch.fcgi", page_params)
        page_ids = page_data.get("esearchresult", {}).get("idlist", [])

        if not page_ids:
            break

        all_ids.extend(page_ids)
        retstart += len(page_ids)
        logger.debug("search_pmc page retstart=%d fetched=%d total_so_far=%d",
                     retstart - len(page_ids), len(page_ids), len(all_ids))

    logger.info("search_pmc query=%r fetched %d / %d IDs", query, len(all_ids), total)
    return [f"PMC{i}" for i in all_ids]


def fetch_fulltext(pmc_id: str, ncbi_api_key: str = "") -> str | None:
    """Fetch full-text XML for a single PMC article. Returns None on permanent failure."""
    numeric_id = pmc_id[3:] if pmc_id.upper().startswith("PMC") else pmc_id
    params: dict = {
        "db": "pmc",
        "id": numeric_id,
        "rettype": "xml",
        "retmode": "xml",
    }
    if ncbi_api_key:
        params["api_key"] = ncbi_api_key

    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(f"{_BASE_URL}/efetch.fcgi", params=params, timeout=30)
            if resp.status_code == 200:
                xml = resp.text
                if "<error>" in xml.lower() or len(xml.strip()) < 100:
                    logger.warning("fetch_fulltext %s empty/error response", pmc_id)
                    return None
                return xml
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = _RETRY_BASE ** attempt
                logger.warning(
                    "fetch_fulltext %s status=%d, retry in %.1fs",
                    pmc_id, resp.status_code, wait,
                )
                time.sleep(wait)
                continue
            logger.warning(
                "fetch_fulltext %s status=%d — permanent failure", pmc_id, resp.status_code
            )
            return None
        except requests.RequestException as exc:
            wait = _RETRY_BASE ** attempt
            logger.warning("fetch_fulltext %s exc=%s, retry in %.1fs", pmc_id, exc, wait)
            time.sleep(wait)

    logger.error("fetch_fulltext %s failed after %d attempts", pmc_id, _MAX_RETRIES)
    return None


def fetch_batch(
    pmc_ids: list[str],
    *,
    workers: int = 4,
    ncbi_api_key: str = "",
    rate: float | None = None,
) -> Iterator[tuple[str, str | None]]:
    """Yield (pmc_id, xml_str | None) pairs, fetching up to `workers` at a time."""
    interval = 1.0 / (rate or (10.0 if ncbi_api_key else float(_DEFAULT_RATE)))

    def _fetch(pmc_id: str) -> tuple[str, str | None]:
        time.sleep(interval)
        return pmc_id, fetch_fulltext(pmc_id, ncbi_api_key=ncbi_api_key)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, pid): pid for pid in pmc_ids}
        for fut in as_completed(futures):
            yield fut.result()


# ── Private helpers ───────────────────────────────────────────────────────────

def _get_json(url: str, params: dict) -> dict:
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            wait = _RETRY_BASE ** attempt
            logger.warning("_get_json %s exc=%s, retry in %.1fs", url, exc, wait)
            time.sleep(wait)
    return {}
