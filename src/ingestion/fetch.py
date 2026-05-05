"""
PubMed Central (PMC) fetching module.

Replaces `data_collection_scaled.py` with a cleaner, async-capable interface.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Iterator

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_DEFAULT_RATE = 3    # req/sec without API key
_MAX_RETRIES = 6
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

    NCBI's esearch hard-caps retstart at 9,998 per query. For topics with more
    than 9,999 results the function automatically splits the search into
    year-by-year windows so each window stays within the cap, then deduplicates
    and returns up to max_results IDs.
    """
    # ── Step 1: get total count for the full date range ───────────────────
    count_params: dict = {
        "db": "pmc",
        "term": query,
        "retmax": 0,
        "retmode": "json",
    }
    if date_range:
        count_params["mindate"] = date_range[0]
        count_params["maxdate"] = date_range[1]
        count_params["datetype"] = "pdat"
    if ncbi_api_key:
        count_params["api_key"] = ncbi_api_key

    count_data = _get_json(f"{_BASE_URL}/esearch.fcgi", count_params)
    total = int(count_data.get("esearchresult", {}).get("count", 0))
    logger.info("search_pmc query=%r total_on_server=%d", query, total)

    # ── Step 2: choose strategy ───────────────────────────────────────────
    if total <= _PAGE_SIZE:
        # Under the cap — single fetch is sufficient.
        windows = [date_range]
    else:
        # Over the cap — split into year-by-year windows.
        start_year = int((date_range[0] if date_range else "2000/01/01").split("/")[0])
        end_year = date.today().year
        windows = [
            (f"{y}/01/01", f"{y}/12/31")
            for y in range(start_year, end_year + 1)
        ]
        logger.info(
            "search_pmc: total %d > 9999 cap — splitting into %d yearly windows",
            total, len(windows),
        )

    # ── Step 3: fetch each window ─────────────────────────────────────────
    seen: set[str] = set()
    all_ids: list[str] = []

    for win_start, win_end in windows:
        if len(all_ids) >= max_results:
            break

        params: dict = {
            "db": "pmc",
            "term": query,
            "usehistory": "y",
            "retmax": 0,
            "retmode": "json",
            "datetype": "pdat",
            "mindate": win_start,
            "maxdate": win_end,
        }
        if ncbi_api_key:
            params["api_key"] = ncbi_api_key

        data = _get_json(f"{_BASE_URL}/esearch.fcgi", params)
        result = data.get("esearchresult", {})
        win_total = int(result.get("count", 0))
        web_env = result.get("webenv", "")
        query_key = result.get("querykey", "")

        if not win_total:
            continue

        if not web_env or not query_key:
            for pid in result.get("idlist", []):
                if pid not in seen and len(all_ids) < max_results:
                    seen.add(pid)
                    all_ids.append(pid)
            continue

        # Paginate within this window (safe: each window is ≤ 9,999)
        retstart = 0
        while retstart < win_total and len(all_ids) < max_results:
            page_size = min(_PAGE_SIZE, win_total - retstart, max_results - len(all_ids))
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

            for pid in page_ids:
                if pid not in seen and len(all_ids) < max_results:
                    seen.add(pid)
                    all_ids.append(pid)
            retstart += len(page_ids)

        logger.debug("search_pmc window=%s/%s win_total=%d collected_so_far=%d",
                     win_start, win_end, win_total, len(all_ids))

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
            try:
                return resp.json()
            except ValueError:
                # NCBI occasionally embeds control characters that invalidate JSON.
                # Strip them and retry the parse before giving up on this attempt.
                sanitized = re.sub(r"[\x00-\x1f\x7f]", "", resp.text)
                return json.loads(sanitized)
        except (requests.RequestException, ValueError) as exc:
            wait = _RETRY_BASE ** attempt
            logger.warning("_get_json %s exc=%s, retry in %.1fs", url, exc, wait)
            time.sleep(wait)
    return {}
