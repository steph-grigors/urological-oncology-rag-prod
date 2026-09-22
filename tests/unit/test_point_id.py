"""
Tests that one chunk has exactly one point id.

Two derivations existed and disagreed:

    src/db/vector_store.py   uuid5(NAMESPACE_DNS, chunk_id)
    src/ingestion/embed.py   UUID(bytes=md5(chunk_id))

A Qdrant upsert replaces by id, so a chunk written through one path and later
re-written through the other would be stored twice rather than updated,
silently doubling it in retrieval. Production's 713,883 points were all
written through the ingestion path, so that derivation is the one kept.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest

from src.db.point_id import chunk_point_id
from src.db.vector_store import _chunk_uuid
from src.ingestion.embed import _stable_uuid

SAMPLE = "PMC1234567_results_003"


def _production_derivation(chunk_id: str) -> str:
    """What the ingestion path wrote for every point now in production."""
    return str(uuid.UUID(bytes=hashlib.md5(chunk_id.encode()).digest()))


class TestOneIdPerChunk:
    def test_both_call_sites_agree(self):
        assert _chunk_uuid(SAMPLE) == _stable_uuid(SAMPLE) == chunk_point_id(SAMPLE)

    @pytest.mark.parametrize("chunk_id", [
        "PMC1_abstract_000", "PMC999999_discussion_042", "x", "PMC1_results_001",
    ])
    def test_agreement_holds_across_inputs(self, chunk_id):
        assert _chunk_uuid(chunk_id) == _stable_uuid(chunk_id)


class TestExistingPointsAreNotOrphaned:
    """The whole point of choosing the ingestion derivation. If this fails,
    every one of the 713,883 points in production has been orphaned and the
    corpus needs a full re-embed."""

    @pytest.mark.parametrize("chunk_id", [
        "PMC1234567_results_003", "PMC42_methods_000", "PMC7654321_conclusion_017",
    ])
    def test_id_matches_what_production_holds(self, chunk_id):
        assert chunk_point_id(chunk_id) == _production_derivation(chunk_id)

    def test_it_is_not_the_old_uuid5_derivation(self):
        assert chunk_point_id(SAMPLE) != str(uuid.uuid5(uuid.NAMESPACE_DNS, SAMPLE))


class TestIdProperties:
    def test_stable_across_calls(self):
        assert chunk_point_id(SAMPLE) == chunk_point_id(SAMPLE)

    def test_distinct_chunks_get_distinct_ids(self):
        assert chunk_point_id("PMC1_a_000") != chunk_point_id("PMC1_a_001")

    def test_is_a_valid_uuid_string(self):
        uuid.UUID(chunk_point_id(SAMPLE))

    def test_module_pulls_in_no_heavy_dependency(self):
        """point_id is imported by the ingestion path, which must not require
        qdrant_client just to compute an id."""
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[2] / "src" / "db" / "point_id.py"
        tree = ast.parse(src.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported <= {"hashlib", "uuid", "__future__"}, imported
