"""
Tests for the metadata cache's concurrency and durability.

Every call used to read the whole cache file, add one entry to its own copy,
and write the whole file back. The pipeline runs extraction from a
ThreadPoolExecutor with two workers, so two interleaving calls lost one of the
two entries every time it happened. The file was also re-read and rewritten in
full on each call, which is quadratic I/O across a run.

Production evidence, 2026-09-20: the cache on the VPS holds 655 entries after
ingesting tens of thousands of papers.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from src.ingestion import extract_metadata as em


@pytest.fixture(autouse=True)
def _clean_cache():
    em._reset_metadata_cache()
    yield
    em._reset_metadata_cache()


def _openai_double():
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({
            "study_design": "rct",
            "sample_size": 100,
            "primary_outcome": "overall survival",
            "intervention": "enzalutamide",
            "comparator": "placebo",
        })))]
    )
    return client


class TestConcurrentWritesAllSurvive:
    def test_parallel_extraction_loses_no_entry(self, tmp_path):
        """The core regression. Two workers, many papers, every result kept."""
        cache_path = str(tmp_path / "metadata_cache.json")
        client = _openai_double()
        n = 60

        def _one(i: int):
            return em.extract_metadata(
                pmid=f"pmid-{i}",
                abstract="A randomised trial of enzalutamide.",
                openai_client=client,
                cache_path=cache_path,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(_one, range(n)))

        em.flush_metadata_cache(cache_path)
        on_disk = json.loads(open(cache_path).read())
        assert len(on_disk) == n, f"expected {n} entries, found {len(on_disk)}"
        assert {f"pmid-{i}" for i in range(n)} == set(on_disk)

    def test_high_contention_still_loses_nothing(self, tmp_path):
        cache_path = str(tmp_path / "c.json")
        client = _openai_double()
        barrier = threading.Barrier(8)

        def _one(i: int):
            barrier.wait()
            em.extract_metadata(f"p{i}", "abstract text", client, cache_path=cache_path)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_one, range(8)))

        em.flush_metadata_cache(cache_path)
        assert len(json.loads(open(cache_path).read())) == 8


class TestCacheStillWorks:
    def test_a_hit_skips_the_api_call(self, tmp_path):
        cache_path = str(tmp_path / "c.json")
        client = _openai_double()
        em.extract_metadata("p1", "abstract", client, cache_path=cache_path)
        em.extract_metadata("p1", "abstract", client, cache_path=cache_path)
        assert client.chat.completions.create.call_count == 1

    def test_an_existing_file_is_read(self, tmp_path):
        cache_path = tmp_path / "c.json"
        cache_path.write_text(json.dumps({
            "p1": {
                "pmid": "p1", "study_design": "meta_analysis", "sample_size": 5,
                "primary_outcome": None, "intervention": None, "comparator": None,
                "extraction_failed": False, "extraction_model": "gpt-4o-mini",
            }
        }))
        client = _openai_double()
        result = em.extract_metadata("p1", "abstract", client, cache_path=str(cache_path))
        assert result.study_design == "meta_analysis"
        client.chat.completions.create.assert_not_called()

    def test_a_corrupt_file_is_treated_as_empty(self, tmp_path):
        cache_path = tmp_path / "c.json"
        cache_path.write_text("{ not json")
        client = _openai_double()
        result = em.extract_metadata("p1", "abstract", client, cache_path=str(cache_path))
        assert result.study_design == "rct"


class TestDurability:
    def test_flush_is_atomic_leaving_no_temp_file(self, tmp_path):
        cache_path = str(tmp_path / "c.json")
        client = _openai_double()
        em.extract_metadata("p1", "abstract", client, cache_path=cache_path)
        em.flush_metadata_cache(cache_path)
        leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []

    def test_flush_on_a_clean_cache_is_a_noop(self, tmp_path):
        cache_path = str(tmp_path / "c.json")
        em.flush_metadata_cache(cache_path)          # nothing loaded yet
        em.flush_metadata_cache(cache_path)

    def test_extractor_exposes_flush(self, tmp_path):
        cache_path = str(tmp_path / "c.json")
        extractor = em.MetadataExtractor(_openai_double(), cache_path=cache_path)
        paper = MagicMock(pmid="p1", abstract="A randomised trial.")
        extractor.extract(paper)
        extractor.flush()
        assert "p1" in json.loads(open(cache_path).read())

    def test_pipeline_flushes_at_batch_boundaries_and_at_the_end(self):
        import inspect

        from src.ingestion import pipeline

        src = inspect.getsource(pipeline.run_ingestion)
        assert src.count("meta_extractor.flush()") >= 2, (
            "expected a flush after the parallel extraction block and at end of run"
        )
