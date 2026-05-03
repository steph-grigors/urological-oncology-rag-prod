"""
PMC XML parsing and section detection.

Takes raw PubMed Central JATS XML strings and produces structured
ParsedPaper objects with clean, normalised section text.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional


# ── Canonical section labels ──────────────────────────────────────────────────

# Checked in order — multi-word phrases come before single keywords so
# "materials and methods" matches before bare "materials".
_SECTION_MAP: list[tuple[str, str]] = [
    # ── Methods variants ──────────────────────────────────────────────────
    ("materials and methods", "methods"),
    ("patients and methods", "methods"),
    ("subjects and methods", "methods"),
    ("participants and methods", "methods"),
    ("study design and methods", "methods"),
    ("study population", "methods"),
    ("study design", "methods"),
    ("statistical analysis", "methods"),
    ("statistical methods", "methods"),
    ("data collection", "methods"),
    ("data analysis", "methods"),
    ("experimental procedures", "methods"),
    # ── Results variants ──────────────────────────────────────────────────
    ("study outcomes", "results"),
    ("clinical outcomes", "results"),
    # ── Conclusion variants ───────────────────────────────────────────────
    ("concluding remarks", "conclusion"),
    # ── Single-word keywords (order matters within this group) ─────────────
    ("abstract", "abstract"),
    ("summary", "abstract"),
    ("introduction", "introduction"),
    ("background", "introduction"),
    ("objective", "introduction"),
    ("rationale", "introduction"),
    ("methods", "methods"),
    ("method", "methods"),
    ("methodology", "methods"),
    ("materials", "methods"),
    ("participants", "methods"),
    ("subjects", "methods"),
    ("patients", "methods"),
    ("protocol", "methods"),
    ("procedure", "methods"),
    ("results", "results"),
    ("findings", "results"),
    ("outcomes", "results"),
    ("outcome", "results"),
    ("discussion", "discussion"),
    ("conclusion", "conclusion"),
    ("implications", "conclusion"),
    # ── Skip sections (indexed entirely) ──────────────────────────────────
    ("references", "_skip"),
    ("bibliography", "_skip"),
    ("acknowledgement", "_skip"),
    ("acknowledgment", "_skip"),
    ("funding", "_skip"),
    ("conflict of interest", "_skip"),
    ("conflicts of interest", "_skip"),
    ("declaration", "_skip"),
    ("competing interest", "_skip"),
    ("abbreviation", "_skip"),
    ("supplementary", "_skip"),
    ("supplemental", "_skip"),
    ("author contribution", "_skip"),
    ("ethics statement", "_skip"),
    ("data availability", "_skip"),
]


@dataclass
class Section:
    name: str          # canonical: abstract | introduction | methods | results |
                       #            discussion | conclusion | other
    raw_name: str      # original heading text from XML
    content: str       # cleaned plain text (or table/figure text)
    section_type: str  # "text" | "table" | "figure_caption"


@dataclass
class ParsedPaper:
    pmc_id: str
    pmid: str
    doi: str
    title: str
    abstract: str
    journal: str
    year: Optional[int]
    authors: list[str]
    sections: list[Section] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def parse_paper(xml_str: str) -> Optional[ParsedPaper]:
    """
    Parse a PMC JATS XML string into a ParsedPaper.

    Returns None if the XML is malformed, has no title, or lacks a <front>
    element with the minimum required metadata.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return None

    article = root.find(".//article") or root
    front = article.find(".//front")
    if front is None:
        return None

    title = _clean(_get_text(front.find(".//article-title")))
    if not title:
        return None

    abstract_elem = front.find(".//abstract")
    abstract = _clean(_get_text(abstract_elem)) if abstract_elem is not None else ""

    journal = _clean(_get_text(front.find(".//journal-title")))

    pub_date = (
        front.find(".//pub-date[@pub-type='epub']")
        or front.find(".//pub-date[@pub-type='ppub']")
        or front.find(".//pub-date")
    )
    year = _extract_year(pub_date)

    doi = _get_text(front.find(".//article-id[@pub-id-type='doi']")).strip()
    pmid = _get_text(front.find(".//article-id[@pub-id-type='pmid']")).strip()

    pmc_id = (
        _get_text(front.find(".//article-id[@pub-id-type='pmc']")).strip()
        or _get_text(front.find(".//article-id[@pub-id-type='pmcid']")).strip()
    )

    authors: list[str] = []
    for contrib in front.findall(".//contrib[@contrib-type='author']"):
        surname = _clean(_get_text(contrib.find(".//surname")))
        given = _clean(_get_text(contrib.find(".//given-names")))
        if surname:
            authors.append(f"{surname} {given}".strip() if given else surname)

    sections: list[Section] = []

    # Abstract is always the first section when present
    if abstract:
        sections.append(Section(
            name="abstract",
            raw_name="Abstract",
            content=abstract,
            section_type="text",
        ))

    body = article.find(".//body")
    if body is not None:
        _walk(body, sections, parent_canonical="other")

    return ParsedPaper(
        pmc_id=pmc_id,
        pmid=pmid,
        doi=doi,
        title=title,
        abstract=abstract,
        journal=journal,
        year=year,
        authors=authors,
        sections=sections,
    )


def normalize_section(heading: str) -> str:
    """
    Map a raw section heading to a canonical label.

    Returns one of:
        abstract | introduction | methods | results | discussion |
        conclusion | other | _skip

    "_skip" means the section should be excluded from the index entirely
    (references, acknowledgements, funding, etc.).
    """
    if not heading:
        return "other"
    lower = heading.lower().strip()
    for keyword, canonical in _SECTION_MAP:
        if keyword in lower:
            return canonical
    return "other"


# ── Private helpers ────────────────────────────────────────────────────────────

def _walk(element: ET.Element, sections: list[Section], parent_canonical: str) -> None:
    """Recursively extract sections from a <body> or <sec> element."""
    for child in element:
        tag = _local(child)

        if tag != "sec":
            continue

        title_elem = child.find("title")
        raw_name = _clean(_get_text(title_elem)) if title_elem is not None else ""
        canonical = normalize_section(raw_name)

        # Hard-skip references and boilerplate
        if canonical == "_skip":
            continue

        # Inherit parent if heading is unrecognised
        if canonical == "other" and parent_canonical not in ("other", "_skip"):
            canonical = parent_canonical

        # ── Direct paragraph text ──────────────────────────────────────────
        para_parts = [
            _clean(_get_text(p))
            for p in child.findall("p")
            if _clean(_get_text(p))
        ]
        # ── List items (eligibility criteria, endpoints, adverse events) ───
        for lst in child.findall("list"):
            items = [
                _clean(_get_text(item))
                for item in lst.findall(".//list-item")
                if _clean(_get_text(item))
            ]
            if items:
                para_parts.append("; ".join(items))

        if para_parts:
            sections.append(Section(
                name=canonical,
                raw_name=raw_name or canonical,
                content=" ".join(para_parts),
                section_type="text",
            ))

        # ── Tables ────────────────────────────────────────────────────────
        for tw in child.findall("table-wrap"):
            text = _table_text(tw)
            if text:
                sections.append(Section(
                    name=canonical,
                    raw_name=raw_name or canonical,
                    content=text,
                    section_type="table",
                ))

        # ── Figure captions ───────────────────────────────────────────────
        for fig in child.findall("fig"):
            text = _figure_caption(fig)
            if text:
                sections.append(Section(
                    name=canonical,
                    raw_name=raw_name or canonical,
                    content=text,
                    section_type="figure_caption",
                ))

        # Recurse into sub-sections
        _walk(child, sections, parent_canonical=canonical)


def _table_text(table_wrap: ET.Element) -> str:
    """Render a <table-wrap> element as plain text rows."""
    parts: list[str] = []

    label = table_wrap.find("label")
    if label is not None:
        parts.append(f"[{_clean(_get_text(label))}]")

    caption = table_wrap.find(".//caption")
    if caption is not None:
        cap = _clean(_get_text(caption))
        if cap:
            parts.append(cap)

    for row in table_wrap.findall(".//thead/tr"):
        cells = [_clean(_get_text(c)) for c in row if _local(c) in ("th", "td")]
        if any(cells):
            parts.append(" | ".join(cells))

    for row in table_wrap.findall(".//tbody/tr"):
        cells = [_clean(_get_text(c)) for c in row if _local(c) in ("td", "th")]
        if any(cells):
            parts.append(" | ".join(cells))

    return "\n".join(parts)


def _figure_caption(fig: ET.Element) -> str:
    """Extract label + caption text from a <fig> element."""
    parts: list[str] = []
    label = fig.find("label")
    if label is not None:
        parts.append(_clean(_get_text(label)))
    caption = fig.find("caption")
    if caption is not None:
        cap = _clean(_get_text(caption))
        if cap:
            parts.append(cap)
    return ": ".join(p for p in parts if p)


def _get_text(element: Optional[ET.Element]) -> str:
    """Recursively collect all text content from an XML element."""
    if element is None:
        return ""
    parts: list[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        parts.append(_get_text(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(parts)


def _clean(text: str) -> str:
    """Remove citation markers and collapse whitespace."""
    # Inline citation markers: [1], [1,2], [1-3], [1, 2], [1–3]
    text = re.sub(r"\[\d+(?:[,\s–\-]\d+)*\]", "", text)
    # Collapse all whitespace to a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _local(element: ET.Element) -> str:
    """Tag name without XML namespace prefix."""
    tag = element.tag
    return tag.split("}")[-1] if "}" in tag else tag


def _extract_year(pub_date: Optional[ET.Element]) -> Optional[int]:
    if pub_date is None:
        return None
    year_elem = pub_date.find("year")
    if year_elem is None:
        return None
    try:
        return int(_get_text(year_elem).strip())
    except ValueError:
        return None
