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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from openai import OpenAI


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
    cache = _load_cache(cache_path)

    if pmid and pmid in cache:
        cached = cache[pmid]
        return ExtractionResult(**cached)

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


def _load_cache(cache_path: str) -> dict:
    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_to_cache(
    cache: dict,
    pmid: str,
    result: ExtractionResult,
    cache_path: str,
) -> None:
    if not pmid:
        return
    cache[pmid] = asdict(result)
    try:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2, ensure_ascii=False)
    except OSError:
        pass  # Cache writes are non-fatal


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
