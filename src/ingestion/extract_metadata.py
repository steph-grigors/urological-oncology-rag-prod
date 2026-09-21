"""
LLM-based study design metadata extraction with a local JSON cache.

Extracts three fields that cannot be reliably parsed from XML alone:
  - study_design  (classified into 6 canonical categories)
  - sample_size   (integer or null)
  - primary_outcome (one sentence or null)

Uses OpenAI structured outputs (response_format with JSON schema).
Requires openai >= 1.40.0; set OPENAI_API_KEY before use.

Cache:
    Results are persisted to data/metadata_cache.json keyed by pmid so
    re-runs skip the LLM call for already-processed papers.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# ── Cache concurrency ────────────────────────────────────────────────────────
# The pipeline extracts metadata from a ThreadPoolExecutor (_META_WORKERS = 2).
# Each call used to read the whole cache file, add one entry to its own copy,
# and write the whole file back. Two threads interleaving that lose one of the
# two entries, every time it happens, and the file was also re-read and
# rewritten in full on every single call -- quadratic I/O over a run.
#
# Production evidence: the cache on the VPS holds 655 entries after ingesting
# tens of thousands of papers.
#
# The cache is now held in memory, guarded by a lock, and flushed to disk every
# _CACHE_FLUSH_EVERY new entries plus on an explicit flush. Losing the last few
# entries to a crash is acceptable: the cache only saves re-extraction cost, it
# is never a source of truth, and every value it holds also goes into the chunk
# payloads at ingest time.

_CACHE_LOCK = threading.Lock()
_CACHE_FLUSH_EVERY = 50
_cache_state: dict[str, dict] = {}      # cache_path -> {pmid: record}
_cache_dirty: dict[str, int] = {}       # cache_path -> unflushed entry count


# ── Study design taxonomy ─────────────────────────────────────────────────────

STUDY_DESIGN_OPTIONS: tuple[str, ...] = (
    "rct",
    "meta_analysis",
    "cohort",
    "case_report",
    "review",
    "unknown",
)

# JSON schema used in the OpenAI structured-output call
_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "study_design": {
            "type": "string",
            "enum": list(STUDY_DESIGN_OPTIONS),
        },
        "sample_size": {
            "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}],
        },
        "primary_outcome": {
            "anyOf": [{"type": "string", "maxLength": 300}, {"type": "null"}],
        },
        "intervention": {
            "anyOf": [{"type": "string", "maxLength": 200}, {"type": "null"}],
        },
        "comparator": {
            "anyOf": [{"type": "string", "maxLength": 200}, {"type": "null"}],
        },
    },
    "required": ["study_design", "sample_size", "primary_outcome", "intervention", "comparator"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """\
You are a clinical evidence analyst. Extract structured information from \
a research paper abstract.

Classification rules for study_design:
  rct          — randomised/randomized controlled trial (must say "random")
  meta_analysis — meta-analysis or systematic review with pooled statistics
  cohort       — prospective or retrospective cohort, observational study,
                 registry study, case-control study
  case_report  — case report or case series with fewer than 10 patients
  review       — narrative review, systematic review without meta-analysis,
                 scoping review, guideline
  unknown      — type cannot be determined from this abstract

Extraction rules:
  • Extract ONLY information explicitly stated. Do NOT infer or assume.
  • sample_size: total enrolled patients/participants as an integer.
    Return null if not stated or ambiguous.
  • primary_outcome: the primary endpoint in one concise sentence.
    Return null if not stated.
  • intervention: the main treatment, drug, or procedure under study (e.g.
    "enzalutamide 160 mg/day", "radical cystectomy", "pembrolizumab").
    Return null if not a comparative or interventional study.
  • comparator: what the intervention is compared against (e.g. "placebo",
    "standard of care", "abiraterone"). Return null if no comparator stated.
"""

_USER_TEMPLATE = """\
Abstract:
{abstract}

Extract: study_design, sample_size, primary_outcome, intervention, comparator.\
"""


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    pmid: str
    study_design: str
    sample_size: Optional[int]
    primary_outcome: Optional[str]
    intervention: Optional[str] = None
    comparator: Optional[str] = None
    extraction_failed: bool = False
    extraction_model: str = ""


# ── Public API ────────────────────────────────────────────────────────────────

def extract_metadata(
    pmid: str,
    abstract: str,
    openai_client: OpenAI,
    cache_path: str = "data/metadata_cache.json",
    model: str = "gpt-4o-mini",
) -> ExtractionResult:
    """
    Return study design, sample size, and primary outcome for a paper.

    Checks the local cache first. On a cache miss, calls the OpenAI API and
    writes the result back to the cache. Failures are cached as
    extraction_failed=True so they are not retried on every run.
    """
    with _CACHE_LOCK:
        cached = _load_cache(cache_path).get(pmid) if pmid else None
    if cached is not None:
        return ExtractionResult(**cached)
    cache = {}

    if not abstract.strip():
        result = ExtractionResult(
            pmid=pmid,
            study_design="unknown",
            sample_size=None,
            primary_outcome=None,
            extraction_failed=True,
            extraction_model=model,
        )
        _save_to_cache(cache, pmid, result, cache_path)
        return result

    result = _call_llm(pmid, abstract, openai_client, model)
    _save_to_cache(cache, pmid, result, cache_path)
    return result


# ── Private helpers ───────────────────────────────────────────────────────────

def _call_llm(
    pmid: str,
    abstract: str,
    client: OpenAI,
    model: str,
) -> ExtractionResult:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _USER_TEMPLATE.format(
                    abstract=abstract[:2000]
                )},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "medical_study_metadata",
                    "strict": True,
                    "schema": _OUTPUT_SCHEMA,
                },
            },
            temperature=0.0,
            max_tokens=150,
        )

        raw = response.choices[0].message.content or "{}"
        data: dict = json.loads(raw)

        return ExtractionResult(
            pmid=pmid,
            study_design=_valid_design(data.get("study_design")),
            sample_size=_to_int(data.get("sample_size")),
            primary_outcome=_trim(data.get("primary_outcome"), 300),
            intervention=_trim(data.get("intervention"), 200),
            comparator=_trim(data.get("comparator"), 200),
            extraction_failed=False,
            extraction_model=model,
        )

    except Exception:
        return ExtractionResult(
            pmid=pmid,
            study_design="unknown",
            sample_size=None,
            primary_outcome=None,
            intervention=None,
            comparator=None,
            extraction_failed=True,
            extraction_model=model,
        )


def _read_cache_file(cache_path: str) -> dict:
    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _load_cache(cache_path: str) -> dict:
    """Return the in-memory cache for `cache_path`, reading the file once.

    Callers must hold _CACHE_LOCK, or treat the result as read-only.
    """
    if cache_path not in _cache_state:
        _cache_state[cache_path] = _read_cache_file(cache_path)
        _cache_dirty[cache_path] = 0
    return _cache_state[cache_path]


def _write_cache_file(cache_path: str, cache: dict) -> None:
    """Write the cache atomically. Caller holds _CACHE_LOCK."""
    try:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        _cache_dirty[cache_path] = 0
    except OSError as exc:
        logger.warning("Could not write metadata cache %s: %s", cache_path, exc)


def flush_metadata_cache(cache_path: str = "data/metadata_cache.json") -> None:
    """Persist any unflushed cache entries. Safe to call at any time."""
    with _CACHE_LOCK:
        cache = _cache_state.get(cache_path)
        if cache is not None and _cache_dirty.get(cache_path):
            _write_cache_file(cache_path, cache)


def _reset_metadata_cache() -> None:
    """Drop the in-memory cache. For tests."""
    with _CACHE_LOCK:
        _cache_state.clear()
        _cache_dirty.clear()


def _save_to_cache(
    cache: dict,
    pmid: str,
    result: ExtractionResult,
    cache_path: str,
) -> None:
    """Record one result. `cache` is accepted for call-site compatibility but
    the authoritative store is the locked in-memory one."""
    if not pmid:
        return
    with _CACHE_LOCK:
        live = _load_cache(cache_path)
        live[pmid] = asdict(result)
        if cache is not live:
            cache[pmid] = live[pmid]
        _cache_dirty[cache_path] = _cache_dirty.get(cache_path, 0) + 1
        if _cache_dirty[cache_path] >= _CACHE_FLUSH_EVERY:
            _write_cache_file(cache_path, live)


def _valid_design(value: object) -> str:
    return value if isinstance(value, str) and value in STUDY_DESIGN_OPTIONS else "unknown"


def _to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        n = int(value)
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


def _trim(value: object, max_len: int) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:max_len] if value else None


# ── Class wrapper expected by pipeline.py ────────────────────────────────────

class MetadataExtractor:
    """Thin class wrapper around extract_metadata() for use by the pipeline."""

    def __init__(
        self,
        openai_client: OpenAI,
        model: str = "gpt-4o-mini",
        cache_path: str = "data/metadata_cache.json",
    ) -> None:
        self._client = openai_client
        self._model = model
        self._cache_path = cache_path

    def flush(self) -> None:
        """Persist any cache entries not yet written to disk.

        Called by the pipeline at batch boundaries and at the end of a run, so
        the buffered tail is never lost to a normal shutdown.
        """
        flush_metadata_cache(self._cache_path)

    def extract(self, paper) -> ExtractionResult:
        """Extract metadata from a ParsedPaper. Falls back to defaults on failure."""
        pmid = getattr(paper, "pmid", "") or ""
        abstract = getattr(paper, "abstract", "") or ""
        return extract_metadata(
            pmid=pmid,
            abstract=abstract,
            openai_client=self._client,
            cache_path=self._cache_path,
            model=self._model,
        )
