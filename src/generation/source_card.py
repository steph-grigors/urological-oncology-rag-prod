"""
Shared "source card" shape used by both /query (SourceCard) and
/treatment-card (sources_detail) — built from a retrieved RankedChunk's
metadata, or from a disclosure placeholder when no chunk backs the entry
(parametric-knowledge fallback).

A single dataclass for the field set means the two call sites can never
drift apart on what a source record looks like.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.reranker import RankedChunk


@dataclass
class SourceDetail:
    chunk_id: str
    title: str
    authors: str
    journal: str
    year: int | None
    study_design: str
    sample_size: int | None
    section: str
    key_finding: str
    pmid: str


def chunk_to_source_detail(chunk: "RankedChunk") -> SourceDetail:
    """Build a SourceDetail from a retrieved chunk's metadata + text."""
    meta = chunk.metadata if hasattr(chunk, "metadata") else {}
    authors_raw = meta.get("authors", [])
    if isinstance(authors_raw, list):
        authors_str = ", ".join(str(a) for a in authors_raw[:3])
        if len(authors_raw) > 3:
            authors_str += " et al."
    else:
        authors_str = str(authors_raw) if authors_raw else ""

    text = chunk.text if hasattr(chunk, "text") else ""
    return SourceDetail(
        chunk_id=getattr(chunk, "chunk_id", ""),
        title=meta.get("title") or "Unknown",
        authors=authors_str,
        journal=meta.get("journal") or "",
        year=meta.get("year"),
        study_design=meta.get("study_design") or "",
        sample_size=meta.get("sample_size"),
        section=meta.get("section") or "",
        key_finding=text[:150],
        pmid=meta.get("pmid") or "",
    )


def disclosure_source_detail(disclosure_text: str) -> SourceDetail:
    """SourceDetail for the parametric-knowledge-fallback disclosure (no real
    chunk backs it) — shares chunk_to_source_detail's field set by
    construction, so the two can never drift apart."""
    return SourceDetail(
        chunk_id="",
        title=disclosure_text,
        authors="",
        journal="",
        year=None,
        study_design="parametric_knowledge",
        sample_size=None,
        section="",
        key_finding="",
        pmid="",
    )
