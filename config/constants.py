"""
Project-wide constants that do not belong in environment configuration.

These values are fixed by design (not operator-configurable) and change only
when the domain model or retrieval architecture changes.
"""

from typing import Final

# ── Cancer topics ────────────────────────────────────────────────────────────

SUPPORTED_TOPICS: Final[list[str]] = [
    "prostate",
    "bladder",
    "kidney",
    "testicular",
]

TOPIC_ALIASES: Final[dict[str, str]] = {
    "renal": "kidney",
    "rcc": "kidney",
    "pca": "prostate",
    "nmibc": "bladder",
    "mibc": "bladder",
    "gct": "testicular",
}

# ── Study design hierarchy (highest → lowest evidence weight) ────────────────
# Used by the reranker and confidence gating to boost evidence-grade signals.

STUDY_DESIGN_HIERARCHY: Final[list[str]] = [
    "systematic review",
    "meta-analysis",
    "randomised controlled trial",
    "rct",
    "randomized controlled trial",
    "prospective cohort",
    "prospective study",
    "retrospective cohort",
    "retrospective study",
    "case-control",
    "cross-sectional",
    "case series",
    "case report",
    "expert opinion",
    "editorial",
    "letter",
]

STUDY_DESIGN_WEIGHTS: Final[dict[str, float]] = {
    "systematic review": 1.0,
    "meta-analysis": 1.0,
    "randomised controlled trial": 0.9,
    "rct": 0.9,
    "randomized controlled trial": 0.9,
    "prospective cohort": 0.75,
    "prospective study": 0.75,
    "retrospective cohort": 0.6,
    "retrospective study": 0.6,
    "case-control": 0.5,
    "cross-sectional": 0.45,
    "case series": 0.3,
    "case report": 0.2,
    "expert opinion": 0.15,
    "editorial": 0.1,
    "letter": 0.1,
}

# ── Section priority for retrieval ───────────────────────────────────────────
# Chunks from high-priority sections receive a retrieval score boost.

SECTION_PRIORITY: Final[dict[str, float]] = {
    "results": 1.0,
    "conclusion": 0.95,
    "conclusions": 0.95,
    "discussion": 0.85,
    "abstract": 0.8,
    "methods": 0.7,
    "introduction": 0.5,
    "background": 0.45,
    "references": 0.0,
    "acknowledgements": 0.0,
    "funding": 0.0,
    "conflict of interest": 0.0,
}

# ── Chunking ─────────────────────────────────────────────────────────────────

MIN_CHUNK_WORDS: Final[int] = 30
MAX_CHUNK_WORDS: Final[int] = 600

# Sections to skip entirely during chunking (no medical signal)
SKIP_SECTIONS: Final[set[str]] = {
    "references",
    "acknowledgements",
    "acknowledgments",
    "funding",
    "conflict of interest",
    "conflicts of interest",
    "abbreviations",
    "supplementary material",
    "supplementary data",
    "author contributions",
    "ethics statement",
    "data availability",
}

# ── Retrieval ─────────────────────────────────────────────────────────────────

BM25_K1: Final[float] = 1.5
BM25_B: Final[float] = 0.75

RRF_K: Final[int] = 60  # Reciprocal Rank Fusion constant

EMBEDDING_DIMENSION: Final[int] = 1536  # text-embedding-3-small

# ── Confidence thresholds ─────────────────────────────────────────────────────

CONFIDENCE_HIGH: Final[float] = 0.75   # Answer with full confidence
CONFIDENCE_LOW: Final[float] = 0.45    # Hedge answer or refuse
CONFIDENCE_REFUSE: Final[float] = 0.2  # Hard refusal — no answer produced

# ── Generation ────────────────────────────────────────────────────────────────

MAX_ANSWER_TOKENS: Final[int] = 2000
GENERATION_TEMPERATURE: Final[float] = 0.1

# Medical disclaimer appended to every answer
MEDICAL_DISCLAIMER: Final[str] = (
    "\n\n*This information is intended for qualified healthcare professionals "
    "and should not replace clinical judgement. Always verify against current "
    "guidelines and individual patient context.*"
)

# ── Audit & compliance ────────────────────────────────────────────────────────

AUDIT_LOG_VERSION: Final[str] = "1.0"
PII_SCRUB_PATTERNS: Final[list[str]] = [
    r"\b\d{3}-\d{2}-\d{4}\b",       # SSN
    r"\b\d{10,16}\b",                 # MRN-like numbers
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # email
]
