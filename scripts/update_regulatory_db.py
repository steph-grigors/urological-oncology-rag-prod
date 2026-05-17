"""
Update data/regulatory_withdrawals.json from two sources:

  1. openFDA drug enforcement API (api.fda.gov) — structured FDA recall and
     voluntary-withdrawal records for oncology drugs.  This is a public REST
     API requiring no authentication.

  2. EMA and FDA web pages fetched via requests + Claude Haiku extraction —
     covers indication-level withdrawals that are announced as press releases
     and regulatory decisions rather than enforcement records.  EMA does not
     offer a public REST API for this data, so page-fetching is the only
     available automated path.

Merge strategy:
  - Existing entries marked "source": "manual" are never overwritten.
  - New entries are keyed by (drug, jurisdiction); duplicates update the
    existing automated entry (fresher data wins).
  - If any source fails, the script logs a warning and continues.
  - The JSON file is only written if at least one entry is found/updated.

Usage:
    ANTHROPIC_API_KEY=<key> python scripts/update_regulatory_db.py

Recommended cadence: weekly (cron) or before each production release.
After running, restart the serving process to reload the updated file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("update_regulatory_db")

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent.parent / "data" / "regulatory_withdrawals.json"
OPENFDA_URL = "https://api.fda.gov/drug/enforcement.json"
REQUEST_TIMEOUT = 30
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; regulatory-db-updater/1.0; "
        "urological-oncology-rag; research use)"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}

# Oncology drugs relevant to urological oncology — queried in openFDA
WATCH_DRUGS = [
    "atezolizumab", "rucaparib", "olaparib", "niraparib", "pembrolizumab",
    "nivolumab", "avelumab", "durvalumab", "cabozantinib", "sunitinib",
    "pazopanib", "axitinib", "enzalutamide", "abiraterone", "darolutamide",
    "apalutamide", "erdafitinib", "sacituzumab", "enfortumab", "lutetium",
]

ONCOLOGY_KEYWORDS = {
    "cancer", "tumor", "tumour", "carcinoma", "malignancy", "oncology",
    "chemotherapy", "immunotherapy", "neoplasm", "bladder", "prostate",
    "renal", "kidney", "urothelial", "testicular",
}

# Pages fetched for Claude-assisted extraction
WEB_SOURCES = [
    {
        "name": "FDA oncology safety notifications",
        "url": (
            "https://www.fda.gov/patients/"
            "hematologyoncology-cancer-approvals-safety-notifications"
        ),
        "jurisdiction_hint": "FDA",
    },
    {
        "name": "EMA cancer medicine news",
        "url": (
            "https://www.ema.europa.eu/en/news?"
            "search_api_views_fulltext=withdrawal+cancer"
        ),
        "jurisdiction_hint": "EMA",
    },
]


# ── HTML stripping ─────────────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Lightweight HTML → plain text converter."""

    _SKIP_TAGS = {"script", "style", "nav", "footer", "head", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._depth = 0
        self._texts: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP_TAGS:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._depth > 0:
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._depth == 0 and data.strip():
            self._texts.append(data.strip())

    def get_text(self) -> str:
        return " ".join(self._texts)


def _strip_html(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    return extractor.get_text()


# ── Source 1: openFDA enforcement API ─────────────────────────────────────────

def query_openfda_enforcement(drug_name: str) -> list[dict]:
    """
    Query openFDA enforcement records for a drug.

    Primarily captures physical recalls (manufacturing, contamination).
    Voluntary withdrawals of specific indications are less common here but
    are included when present.
    """
    params = {
        "search": f'product_description:"{drug_name}"',
        "limit": 20,
    }
    try:
        resp = requests.get(OPENFDA_URL, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 404:
            return []  # No records for this drug
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except requests.HTTPError as exc:
        logger.warning("openFDA HTTP error for %s: %s", drug_name, exc)
        return []
    except Exception as exc:
        logger.warning("openFDA query failed for %s: %s", drug_name, exc)
        return []


def _is_oncology(record: dict) -> bool:
    text = " ".join([
        record.get("product_description", ""),
        record.get("reason_for_recall", ""),
    ]).lower()
    return any(kw in text for kw in ONCOLOGY_KEYWORDS)


def openfda_record_to_entry(drug_name: str, record: dict) -> dict | None:
    """Convert an openFDA enforcement record to our withdrawal entry schema."""
    if not _is_oncology(record):
        return None
    if "voluntary" not in record.get("voluntary_mandated", "").lower():
        return None

    product_desc = record.get("product_description", "")
    reason = record.get("reason_for_recall", "No reason provided")
    date_raw = record.get("recall_initiation_date", "")
    try:
        date = datetime.strptime(date_raw, "%Y%m%d").strftime("%Y-%m")
    except ValueError:
        date = date_raw[:7] if len(date_raw) >= 7 else date_raw

    return {
        "drug": drug_name.lower(),
        "aliases": [],
        "indication": "see product description",
        "indication_keywords": [],
        "jurisdiction": "FDA",
        "status": "withdrawn" if record.get("status") == "Terminated" else "recalled",
        "date": date,
        "warning": (
            f"{drug_name.title()} — FDA voluntary action ({date}): "
            f"{reason[:200].rstrip('.')}. "
            "Verify current regulatory status before recommending."
        ),
        "source": "openfda_api",
        "raw_description": product_desc[:300],
    }


def collect_openfda_entries() -> list[dict]:
    """Run openFDA queries for all watch drugs and return relevant entries."""
    entries: list[dict] = []
    for drug in WATCH_DRUGS:
        logger.info("openFDA query: %s", drug)
        records = query_openfda_enforcement(drug)
        for rec in records:
            entry = openfda_record_to_entry(drug, rec)
            if entry:
                entries.append(entry)
        time.sleep(0.3)  # respect API rate limit (240 req/min for unauthenticated)
    logger.info("openFDA: found %d oncology enforcement entries", len(entries))
    return entries


# ── Source 2: web pages + Claude Haiku extraction ─────────────────────────────

def fetch_page_text(url: str) -> str:
    """Fetch a URL and return stripped plain text (max 10 000 chars)."""
    try:
        resp = requests.get(
            url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True
        )
        resp.raise_for_status()
        return _strip_html(resp.text)[:10_000]
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return ""


def extract_entries_with_claude(
    content: str,
    jurisdiction_hint: str,
    source_name: str,
    api_key: str,
) -> list[dict]:
    """
    Use Claude Haiku with tool_use to extract structured withdrawal entries
    from the plain-text content of a regulatory web page.
    """
    if not content.strip():
        return []

    import anthropic  # import here so the module loads without the SDK installed

    tool = {
        "name": "record_withdrawals",
        "description": (
            "Record regulatory drug withdrawal or suspension entries found in the content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "description": "List of withdrawal entries found. Empty list if none.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "drug": {
                                "type": "string",
                                "description": "Canonical drug name, lowercase.",
                            },
                            "aliases": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Brand names and alternate spellings.",
                            },
                            "indication": {
                                "type": "string",
                                "description": "The specific indication that was withdrawn.",
                            },
                            "indication_keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "2–5 lowercase keywords from the indication "
                                    "for substring matching (e.g. 'urothelial', 'crpc')."
                                ),
                            },
                            "jurisdiction": {
                                "type": "string",
                                "description": "Regulatory body: EMA, FDA, MHRA, etc.",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["withdrawn", "suspended"],
                            },
                            "date": {
                                "type": "string",
                                "description": "YYYY-MM or YYYY.",
                            },
                            "warning": {
                                "type": "string",
                                "description": (
                                    "1–2 sentence clinician-facing warning. Include drug "
                                    "name, jurisdiction, indication, date, and 'Verify "
                                    "current regulatory status before recommending.'"
                                ),
                            },
                        },
                        "required": ["drug", "jurisdiction", "status", "warning"],
                    },
                }
            },
            "required": ["entries"],
        },
    }

    prompt = (
        f"You are a pharmacovigilance assistant reviewing a page from {source_name}.\n\n"
        f"Extract every regulatory withdrawal or suspension of marketing authorisation "
        f"for oncology drugs (cancer medicines). Focus on {jurisdiction_hint} actions "
        f"but include others if found.\n\n"
        f"Only extract WITHDRAWALS or SUSPENSIONS — not approvals, label updates, or "
        f"ongoing reviews.  If nothing was withdrawn or suspended, return an empty list.\n\n"
        f"Page content:\n{content}"
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            tools=[tool],
            tool_choice={"type": "required"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "record_withdrawals":
                raw = block.input.get("entries", [])
                for entry in raw:
                    entry.setdefault("aliases", [])
                    entry.setdefault("indication_keywords", [])
                    entry["source"] = "claude_extraction"
                    entry["drug"] = entry["drug"].lower()
                logger.info(
                    "Claude extracted %d entries from %s", len(raw), source_name
                )
                return raw
        return []
    except Exception as exc:
        logger.warning("Claude extraction failed for %s: %s", source_name, exc)
        return []


def collect_web_entries(api_key: str) -> list[dict]:
    """Fetch configured web sources and extract withdrawal entries with Claude."""
    entries: list[dict] = []
    for source in WEB_SOURCES:
        logger.info("Fetching: %s", source["name"])
        text = fetch_page_text(source["url"])
        if text:
            extracted = extract_entries_with_claude(
                content=text,
                jurisdiction_hint=source["jurisdiction_hint"],
                source_name=source["name"],
                api_key=api_key,
            )
            entries.extend(extracted)
    return entries


# ── Merge & persist ────────────────────────────────────────────────────────────

def merge_entries(existing: list[dict], new_entries: list[dict]) -> list[dict]:
    """
    Merge new entries into existing, deduplicating by (drug, jurisdiction).

    Manual entries ("source": "manual") are never overwritten.
    Automated entries are overwritten by fresher automated data.
    """
    merged: dict[tuple[str, str], dict] = {}
    for entry in existing:
        key = (entry.get("drug", "").lower(), entry.get("jurisdiction", "").upper())
        merged[key] = entry

    protected = 0
    updated = 0
    added = 0
    for entry in new_entries:
        drug = entry.get("drug", "").strip().lower()
        jur = entry.get("jurisdiction", "").strip().upper()
        warning = entry.get("warning", "").strip()
        if not drug or not warning:
            continue
        key = (drug, jur)
        existing_entry = merged.get(key)
        if existing_entry and existing_entry.get("source") == "manual":
            protected += 1
            continue
        if key in merged:
            updated += 1
        else:
            added += 1
        merged[key] = entry

    logger.info(
        "Merge: %d protected (manual), %d updated, %d added",
        protected, updated, added,
    )
    return list(merged.values())


def load_existing() -> list[dict]:
    if not DATA_PATH.exists():
        return []
    try:
        raw = json.loads(DATA_PATH.read_text())
        return raw.get("entries", raw) if isinstance(raw, dict) else raw
    except Exception as exc:
        logger.warning("Could not read existing %s: %s", DATA_PATH, exc)
        return []


def write_output(entries: list[dict]) -> None:
    output = {
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "automated (openFDA API + Claude-assisted web extraction)",
        "entries": entries,
    }
    DATA_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info("Written %d entries to %s", len(entries), DATA_PATH)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set — web extraction skipped; "
            "only openFDA data will be collected."
        )

    existing = load_existing()
    logger.info("Loaded %d existing entries", len(existing))

    all_new: list[dict] = []

    # Source 1: openFDA enforcement API
    all_new.extend(collect_openfda_entries())

    # Source 2: web pages + Claude extraction (skipped if no API key)
    if api_key:
        all_new.extend(collect_web_entries(api_key))
    else:
        logger.info("Skipping web extraction (no API key)")

    if not all_new and not existing:
        logger.error("No entries found from any source and no existing data — aborting")
        sys.exit(1)

    merged = merge_entries(existing, all_new)
    write_output(merged)
    logger.info("Done — %d total entries in %s", len(merged), DATA_PATH)


if __name__ == "__main__":
    main()
