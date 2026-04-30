"""
Section-aware chunking module.

Replaces `data_processing_scaled.py` with a richer chunking strategy that
preserves clinical context across boundaries.

Strategy:
- Each section is chunked independently so chunk boundaries never span
  across two different sections (e.g., Methods text never bleeds into
  Results text).
- Chunk size and overlap are controlled by `config.settings` (CHUNK_SIZE_WORDS,
  CHUNK_OVERLAP_WORDS) so they can be tuned without code changes.
- Very short sections (< MIN_CHUNK_WORDS) are kept as a single chunk;
  sections in SKIP_SECTIONS are dropped entirely.
- Each `Chunk` carries a copy of paper-level metadata plus section-level
  fields so retrieval results are self-contained.
- A `context_prefix` field is populated with the paper title + section name
  so the embedding captures document structure ("Kidney Cancer RCT — Results:
  <chunk text>") — this can be toggled via a flag for ablation studies.

Public API (to be implemented):
    chunk_paper(paper: ParsedPaper, settings: Settings) -> list[Chunk]
        Produce all chunks for a single parsed paper.

    chunk_section(text: str, section_name: str, metadata: ChunkMetadata,
                  chunk_size: int, overlap: int) -> list[Chunk]
        Low-level chunker for a single section's text.

    Chunk(dataclass)
        id: str               # deterministic hash of (pmc_id, section, index)
        text: str             # raw chunk body
        context_prefix: str   # title + section prepended for embedding
        metadata: ChunkMetadata
        chunk_index: int
        total_chunks: int     # total chunks in this section

    ChunkMetadata(dataclass)
        pmc_id, pmid, doi, title, authors, journal, year, topic,
        section_name, section_type, study_design, num_sections
"""
