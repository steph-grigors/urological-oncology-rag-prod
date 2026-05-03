"""
Section-aware chunking module.

Converts a ParsedPaper into a flat list of Chunks, each carrying the full
metadata schema required by the retrieval layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config.constants import MIN_CHUNK_WORDS
from src.ingestion.parse import ParsedPaper, Section


# ── Public constants ──────────────────────────────────────────────────────────

SHORT_SECTION_THRESHOLD: int = 80     # words; shorter sections → single chunk
TABLE_WORD_CAP: int = 500             # words; longer tables are truncated
TABLE_TRUNCATION_NOTE: str = "[Table truncated for length]"

# Maps study_design string to evidence level integer (lower = stronger)
EVIDENCE_LEVELS: dict[str, int] = {
    "meta_analysis": 1,
    "rct":           2,
    "cohort":        3,
    "case_report":   4,
    "review":        5,
    "unknown":       6,
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ChunkMetadata:
    pmid: str
    pmcid: str
    title: str
    authors: list[str]
    journal: str
    year: Optional[int]
    cancer_type: list[str]
    section: str          # canonical section label
    chunk_type: str       # "text" | "table" | "figure_caption"
    chunk_index: int      # global position within the paper
    study_design: str
    sample_size: Optional[int]
    primary_outcome: Optional[str]
    evidence_level: int
    intervention: Optional[str] = None
    comparator: Optional[str] = None


@dataclass
class Chunk:
    id: str              # stable: "{pmcid}_{section}_{chunk_index}"
    text: str
    metadata: ChunkMetadata


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_paper(
    paper: ParsedPaper,
    cancer_type: list[str],
    study_design: str = "unknown",
    sample_size: Optional[int] = None,
    primary_outcome: Optional[str] = None,
    intervention: Optional[str] = None,
    comparator: Optional[str] = None,
    chunk_size: int = 200,
    overlap: int = 30,
) -> list[Chunk]:
    """
    Produce all chunks for a parsed paper.

    chunk_index is a global counter that increases monotonically across every
    section in the paper, so each chunk has a unique position within its paper.

    cancer_type, study_design, sample_size, primary_outcome, and evidence_level
    are paper-level fields that do not come from the XML — they must be supplied
    by the caller (typically from extract_metadata or the ingestion pipeline).
    """
    base = dict(
        pmid=paper.pmid,
        pmcid=paper.pmc_id,
        title=paper.title,
        authors=paper.authors,
        journal=paper.journal,
        year=paper.year,
        cancer_type=cancer_type,
        study_design=study_design,
        sample_size=sample_size,
        primary_outcome=primary_outcome,
        intervention=intervention,
        comparator=comparator,
        evidence_level=EVIDENCE_LEVELS.get(study_design, 6),
    )

    all_chunks: list[Chunk] = []
    global_index = 0

    for section in paper.sections:
        if not section.content.strip():
            continue

        new = chunk_section(
            section=section,
            base_metadata=base,
            start_index=global_index,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(new)
        global_index += len(new)

    return all_chunks


def chunk_section(
    section: Section,
    base_metadata: dict,
    start_index: int,
    chunk_size: int = 200,
    overlap: int = 30,
) -> list[Chunk]:
    """
    Produce chunks for a single Section.

    Rules:
      - Tables: always one chunk, body text capped at TABLE_WORD_CAP words.
      - Figure captions: always one chunk.
      - Text sections with <= SHORT_SECTION_THRESHOLD words: one chunk.
      - Longer text sections: overlapping sliding-window chunks.

    base_metadata must contain all ChunkMetadata fields except section,
    chunk_type, and chunk_index, which are filled in here.
    """
    if not section.content.strip():
        return []

    pmcid = base_metadata["pmcid"]
    chunk_type = _map_chunk_type(section.section_type)

    if section.section_type == "table":
        return _table_chunk(section, base_metadata, start_index, pmcid)

    if section.section_type == "figure_caption":
        return [_single(section.content, section.name, chunk_type,
                        start_index, pmcid, base_metadata)]

    # Text section
    words = section.content.split()
    if not words:
        return []

    # Drop fragments too short to carry useful medical signal
    if len(words) < MIN_CHUNK_WORDS:
        return []

    if len(words) <= SHORT_SECTION_THRESHOLD:
        return [_single(section.content, section.name, chunk_type,
                        start_index, pmcid, base_metadata)]

    return _sliding(words, section.name, chunk_type,
                    start_index, pmcid, base_metadata, chunk_size, overlap)


# ── Private helpers ───────────────────────────────────────────────────────────

def _single(
    text: str,
    section: str,
    chunk_type: str,
    index: int,
    pmcid: str,
    base: dict,
) -> Chunk:
    return Chunk(
        id=_make_id(pmcid, section, index),
        text=text,
        metadata=ChunkMetadata(
            section=section,
            chunk_type=chunk_type,
            chunk_index=index,
            **base,
        ),
    )


def _table_chunk(
    section: Section,
    base: dict,
    index: int,
    pmcid: str,
) -> list[Chunk]:
    words = section.content.split()
    if len(words) > TABLE_WORD_CAP:
        text = " ".join(words[:TABLE_WORD_CAP]) + " " + TABLE_TRUNCATION_NOTE
    else:
        text = section.content

    return [Chunk(
        id=_make_id(pmcid, section.name, index),
        text=text,
        metadata=ChunkMetadata(
            section=section.name,
            chunk_type="table",
            chunk_index=index,
            **base,
        ),
    )]


def _sliding(
    words: list[str],
    section: str,
    chunk_type: str,
    start_index: int,
    pmcid: str,
    base: dict,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Sliding-window chunker with overlap."""
    step = chunk_size - overlap
    chunks: list[Chunk] = []
    pos = 0

    while pos < len(words):
        slice_words = words[pos: pos + chunk_size]
        global_idx = start_index + len(chunks)

        chunks.append(Chunk(
            id=_make_id(pmcid, section, global_idx),
            text=" ".join(slice_words),
            metadata=ChunkMetadata(
                section=section,
                chunk_type=chunk_type,
                chunk_index=global_idx,
                **base,
            ),
        ))

        # Stop once we've consumed all words
        if pos + chunk_size >= len(words):
            break
        pos += step

    return chunks


def _make_id(pmcid: str, section: str, index: int) -> str:
    return f"{pmcid}_{section}_{index}"


def _map_chunk_type(section_type: str) -> str:
    return {"text": "text", "table": "table",
            "figure_caption": "figure_caption"}.get(section_type, "text")
