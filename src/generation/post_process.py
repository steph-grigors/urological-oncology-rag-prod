"""
Post-generation safety checks applied to all generation paths.

Currently implements regulatory withdrawal warnings.  Biomarker eligibility
gating is deferred to the /treatment-card endpoint (Phase 3) where patient
context is available.

The warnings JSON is loaded once at process startup via lru_cache.  To pick
up a fresh file after running scripts/update_regulatory_db.py, restart the
serving process (standard practice for config-file changes).
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ── Data loading ──────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _load_withdrawals() -> tuple[dict, ...]:
    """Load and cache regulatory withdrawal entries from data/regulatory_withdrawals.json."""
    path = _DATA_DIR / "regulatory_withdrawals.json"
    if not path.exists():
        logger.warning(
            "regulatory_withdrawals.json not found at %s — withdrawal checks skipped", path
        )
        return ()
    try:
        raw = json.loads(path.read_text())
        entries = raw.get("entries", raw) if isinstance(raw, dict) else raw
        logger.debug("Loaded %d regulatory withdrawal entries", len(entries))
        return tuple(entries)
    except Exception:
        logger.exception("Failed to load regulatory_withdrawals.json — withdrawal checks skipped")
        return ()


# ── Public API ────────────────────────────────────────────────────────────────

def apply_regulatory_warnings(answer: str) -> str:
    """
    Append regulatory withdrawal warnings to the answer when a withdrawn drug
    is mentioned in the context of its withdrawn indication.

    Fires when both conditions are met:
      1. A drug name or alias appears in the answer (case-insensitive substring)
      2. At least one indication keyword for that entry appears in the answer
         (or the entry has no indication_keywords, in which case it always fires)

    Multiple matching entries produce multiple warning paragraphs.
    The original answer is always returned unchanged if no entries match.
    """
    entries = _load_withdrawals()
    if not entries:
        return answer

    found: list[str] = []
    answer_lower = answer.lower()

    for entry in entries:
        names = [entry.get("drug", "")] + entry.get("aliases", [])
        if not any(n.lower() in answer_lower for n in names if n):
            continue

        indication_kws = entry.get("indication_keywords", [])
        if indication_kws and not any(kw.lower() in answer_lower for kw in indication_kws):
            continue

        warning = entry.get("warning", "")
        if warning:
            found.append(f"⚠️ **Regulatory note:** {warning}")

    if found:
        return answer.rstrip() + "\n\n" + "\n\n".join(found)
    return answer
