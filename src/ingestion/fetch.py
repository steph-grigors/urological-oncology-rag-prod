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

MESH_TERMS: dict[str, str] = {
    "prostate":   '"Prostatic Neoplasms"[MeSH]',
    "bladder":    '"Urinary Bladder Neoplasms"[MeSH]',
    "kidney":     '"Kidney Neoplasms"[MeSH]',
    "testicular": '"Testicular Neoplasms"[MeSH]',
}


def search_pmc(
    query: str,
    max_results: int = 300,
    date_range: tuple[str, str] | None = None,
    ncbi_api_key: str = "",
) -> list[str]:
    """Return a list of PMC IDs matching the query."""
    params: dict = {
        "db": "pmc",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    if date_range:
        params["mindate"] = date_range[0]
        params["maxdate"] = date_range[1]
        params["datetype"] = "pdat"
    if ncbi_api_key:
        params["api_key"] = ncbi_api_key

    data = _get_json(f"{_BASE_URL}/esearch.fcgi", params)
    ids = data.get("esearchresult", {}).get("idlist", [])
    logger.info("search_pmc query=%r found %d IDs", query, len(ids))
    return [f"PMC{i}" for i in ids]


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
