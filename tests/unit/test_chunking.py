"""
Unit tests for src/ingestion/chunk.py.

Tests cover:
    - Short sections (< MIN_CHUNK_WORDS) are kept as a single chunk.
    - Standard section produces correct number of chunks given size/overlap.
    - Overlap content is present at the boundary of adjacent chunks.
    - SKIP_SECTIONS sections produce no chunks.
    - Deterministic chunk IDs: same input → same ID on repeated calls.
    - ChunkMetadata fields are all populated from the ParsedPaper fixture.
    - context_prefix contains both paper title and section name.
    - Maximum chunk word count is never exceeded.
    - Empty section content produces zero chunks (no empty-string chunks).
"""

import pytest


# TODO: import chunk_paper, chunk_section, Chunk, ChunkMetadata from src.ingestion.chunk


class TestChunkSection:
    def test_short_section_single_chunk(self):
        raise NotImplementedError

    def test_standard_section_chunk_count(self):
        raise NotImplementedError

    def test_overlap_content_at_boundary(self):
        raise NotImplementedError

    def test_never_exceeds_max_words(self):
        raise NotImplementedError

    def test_empty_content_produces_no_chunks(self):
        raise NotImplementedError


class TestSkipSections:
    def test_references_section_skipped(self):
        raise NotImplementedError

    def test_acknowledgements_skipped(self):
        raise NotImplementedError


class TestChunkIds:
    def test_deterministic_id(self):
        raise NotImplementedError

    def test_unique_ids_across_sections(self):
        raise NotImplementedError


class TestContextPrefix:
    def test_prefix_contains_title(self):
        raise NotImplementedError

    def test_prefix_contains_section_name(self):
        raise NotImplementedError
