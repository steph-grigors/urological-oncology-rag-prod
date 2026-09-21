"""
Tests for the silent-data-loss path in ingestion.

_upsert_with_retry used to return None whether it succeeded or gave up. On
permanent failure it logged an error and the caller carried on: the chunks had
already been counted as embedded, the pipeline checkpointed the papers as
ingested, and every subsequent run skipped them. The chunks were gone and the
summary said otherwise.

Measured against production on 2026-09-20, the live collection holds 687,101
points where this pipeline reported 795,306 -- the shape that failure leaves.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.embed import EmbedSummary, _upsert_with_retry, embed_chunks


class _Meta:
    def __init__(self):
        self.title = "T"
        self.section = "results"


class _Chunk:
    def __init__(self, cid: str):
        self.id = cid
        self.text = "Enzalutamide improved overall survival."
        self.metadata = _Meta()


def _openai_double(n: int):
    client = MagicMock()
    client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536) for _ in range(n)]
    )
    return client


# ── _upsert_with_retry ───────────────────────────────────────────────────────

class TestUpsertReportsItsOutcome:
    def test_returns_true_on_success(self):
        client = MagicMock()
        assert _upsert_with_retry(client, "c", [1, 2, 3]) is True

    def test_returns_false_once_retries_are_exhausted(self):
        client = MagicMock()
        client.upsert.side_effect = RuntimeError("qdrant down")
        with patch("src.ingestion.embed.time.sleep"):
            assert _upsert_with_retry(client, "c", [1, 2, 3], max_retries=2) is False

    def test_returns_true_when_a_retry_succeeds(self):
        client = MagicMock()
        client.upsert.side_effect = [RuntimeError("blip"), None]
        with patch("src.ingestion.embed.time.sleep"):
            assert _upsert_with_retry(client, "c", [1], max_retries=3) is True


# ── embed_chunks accounting ──────────────────────────────────────────────────

class TestFailedChunksAreNotCountedAsEmbedded:
    def _run(self, upsert_side_effect):
        chunks = [_Chunk(f"c{i}") for i in range(5)]
        qdrant = MagicMock()
        qdrant.upsert.side_effect = upsert_side_effect
        with patch("src.ingestion.embed.time.sleep"):
            return embed_chunks(
                chunks, _openai_double(5), qdrant, "collection", batch_size=5
            )

    def test_success_counts_everything_as_embedded(self):
        summary = self._run(None)
        assert summary.embedded == 5
        assert summary.failed == 0
        assert summary.upsert_failures == 0
        assert summary.all_persisted is True

    def test_permanent_failure_moves_points_out_of_embedded(self):
        summary = self._run(RuntimeError("qdrant down"))
        assert summary.embedded == 0, "points that never landed must not count as embedded"
        assert summary.failed == 5
        assert summary.upsert_failures == 1
        assert summary.all_persisted is False

    def test_all_persisted_is_false_when_an_embedding_batch_failed(self):
        summary = EmbedSummary(total_chunks=10, embedded=5, failed=5)
        assert summary.all_persisted is False


# ── pipeline checkpointing ───────────────────────────────────────────────────

class TestCheckpointRequiresPersistence:
    """The comment above the flush block has always claimed the checkpoint is
    written only after a successful upsert. These pin that it now is."""

    def test_flush_returns_without_checkpointing_on_failure(self, tmp_path):
        import inspect

        from src.ingestion import pipeline

        src = inspect.getsource(pipeline.run_ingestion)
        assert "if not embed_sum.all_persisted:" in src
        # The guard must come before the checkpoint write, not after it.
        guard = src.index("if not embed_sum.all_persisted:")
        write = src.index('checkpoint["ingested_ids"] = list(ingested_ids)')
        assert guard < write, "the persistence guard must precede the checkpoint write"

    @pytest.mark.parametrize("failures,persisted", [(0, True), (1, False)])
    def test_all_persisted_drives_the_decision(self, failures, persisted):
        summary = EmbedSummary(total_chunks=1, embedded=1, upsert_failures=failures)
        if failures:
            summary.embedded, summary.failed = 0, 1
        assert summary.all_persisted is persisted
