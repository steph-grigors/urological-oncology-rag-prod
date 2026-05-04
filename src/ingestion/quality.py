"""
Paper quality scoring and filtering.

Runs on every ParsedPaper immediately after XML parsing, before any chunking
or embedding.  Low-quality papers are logged and their PMC IDs written to
data/ingestion_rejected.json so they can be reviewed or re-fetched later.

Scoring model
─────────────
Each check contributes its weight to a score in [0, 1].  A paper passes when
score >= PASS_SCORE.  The weights are tuned so that a paper *must* have at
least moderate body content (0.35) plus some evidence of findings (0.30) to
reach the threshold — abstract-only or findings-free papers cannot pass.

  Check                  Weight  Rationale
  ─────────────────────  ──────  ──────────────────────────────────────────────
  abstract_present        0.15   Needed for metadata extraction (LLM call)
  abstract_length         0.10   < 50 words → study design classification fails
  body_content            0.35   Core signal; abstract-only papers score ≤ 0.25
  results_or_discussion   0.30   No findings → useless for clinical queries
  section_variety         0.10   Single-section dumps indicate parsing failure
  ─────────────────────  ──────
  Total                   1.00
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

from src.ingestion.parse import ParsedPaper

logger = logging.getLogger(__name__)


# ── Configurable thresholds ───────────────────────────────────────────────────

MIN_ABSTRACT_WORDS: Final[int] = 50
MIN_BODY_WORDS: Final[int] = 300
MIN_DISTINCT_SECTIONS: Final[int] = 2
PASS_SCORE: Final[float] = 0.50

# Canonical section names that indicate the paper contains actual findings
_RESULT_SECTIONS: Final[frozenset[str]] = frozenset({
    "results", "discussion", "conclusion",
})

# Weight of each check — must sum to 1.0
_WEIGHTS: Final[dict[str, float]] = {
    "abstract_present":      0.15,
    "abstract_length":       0.10,
    "body_content":          0.35,
    "results_or_discussion": 0.30,
    "section_variety":       0.10,
}

# Hard blockers: paper auto-fails if any of these checks fail, regardless of score.
# A paper with no findings (results/discussion/conclusion) cannot usefully answer
# clinical questions and must be excluded even if the rest of its content is good.
_REQUIRED_CHECKS: Final[tuple[str, ...]] = ("results_or_discussion",)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class QualityResult:
    passed: bool
    score: float                              # 0.0–1.0; >= PASS_SCORE → passed
    reasons: list[str] = field(default_factory=list)  # one entry per failed check


# ── Public API ────────────────────────────────────────────────────────────────

def score_paper_quality(paper: ParsedPaper) -> QualityResult:
    """
    Score a ParsedPaper and return a QualityResult.

    Checks are evaluated against the parsed content only — no I/O, no LLM
    calls.  The function is pure and safe to call in tight ingestion loops.
    """
    abstract_words = len(paper.abstract.split()) if paper.abstract else 0

    body_sections = [s for s in paper.sections if s.name != "abstract"]
    body_words = sum(len(s.content.split()) for s in body_sections)

    # Exclude "other" from variety count — it's a catch-all, not a real section
    distinct_sections = {s.name for s in body_sections if s.name != "other"}
    has_findings = bool(_RESULT_SECTIONS & distinct_sections)

    checks: dict[str, bool] = {
        "abstract_present":      bool(paper.abstract and paper.abstract.strip()),
        "abstract_length":       abstract_words >= MIN_ABSTRACT_WORDS,
        "body_content":          body_words >= MIN_BODY_WORDS,
        "results_or_discussion": has_findings,
        "section_variety":       len(distinct_sections) >= MIN_DISTINCT_SECTIONS,
    }

    reasons_map: dict[str, str] = {
        "abstract_present":      "Abstract missing",
        "abstract_length":       (
            f"Abstract too short ({abstract_words} words, min {MIN_ABSTRACT_WORDS})"
        ),
        "body_content":          (
            f"Body content too short ({body_words} words, min {MIN_BODY_WORDS})"
        ),
        "results_or_discussion": (
            "No results, discussion, or conclusion section found"
        ),
        "section_variety":       (
            f"Too few distinct body sections "
            f"({len(distinct_sections)}, min {MIN_DISTINCT_SECTIONS})"
        ),
    }

    score = sum(_WEIGHTS[k] for k, passed in checks.items() if passed)
    reasons = [reasons_map[k] for k, passed in checks.items() if not passed]

    required_ok = all(checks[k] for k in _REQUIRED_CHECKS)
    passed = required_ok and score >= PASS_SCORE

    return QualityResult(
        passed=passed,
        score=round(score, 4),
        reasons=reasons,
    )
