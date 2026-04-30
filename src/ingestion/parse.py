"""
PMC XML parsing and section detection.

Takes raw PubMed Central JATS XML strings and produces structured
`ParsedPaper` objects with clean, normalised section text.

Improvements over `data_collection_scaled.py`:
- Typed `ParsedPaper` dataclass instead of raw dicts.
- Section-name normalisation: maps variant headings
  ("Materials and Methods", "Patients and Methods", etc.) to a canonical
  set defined in `config/constants.py`.
- Skips non-informative sections (references, acknowledgements, funding)
  as listed in `SKIP_SECTIONS`.
- Preserves table captions and figure captions as separate section types
  so they can be chunked and indexed independently.
- Strips XML/HTML artefacts, LaTeX fragments, and citation markers
  (e.g., "[1]", "(Smith et al., 2020)") that pollute embedding space.

Public API (to be implemented):
    parse_paper(xml_str: str) -> ParsedPaper | None
        Parse a single PMC XML string. Returns None if the XML is
        malformed or the article lacks a usable body.

    ParsedPaper(dataclass)
        pmc_id, pmid, doi, title, abstract, journal, year, authors,
        sections: list[Section], study_design: str | None

    Section(dataclass)
        name: str            # canonical section name
        raw_name: str        # original heading text from XML
        content: str         # clean plain text
        section_type: Literal["body", "table_caption", "figure_caption", "abstract"]
"""
