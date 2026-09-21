"""
Tests that payload indexes are ensured on an existing collection.

Index creation used to sit inside the "collection is missing" branch of
ensure_collection, so it ran only when QdrantStore created the collection
itself. Production never takes that path: src/ingestion/embed.ensure_collection
creates the collection with vectors and no indexes, and QdrantStore then
attaches to it.

Confirmed against production on 2026-09-20 -- the live collection reported no
payload indexes at all, so every filtered search scanned all 687,101 points.
/treatment-card filters on cancer_type for every request.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.db.vector_store import (
    PAYLOAD_INTEGER_FIELDS,
    PAYLOAD_KEYWORD_FIELDS,
    QdrantStore,
)


def _client(*, collection_exists: bool, indexed_fields=(), get_collection_raises=False):
    client = MagicMock()
    names = [MagicMock()]
    names[0].name = "existing_collection" if collection_exists else "something_else"
    client.get_collections.return_value = MagicMock(collections=names)

    if get_collection_raises:
        client.get_collection.side_effect = RuntimeError("older qdrant")
    else:
        client.get_collection.return_value = MagicMock(
            payload_schema={f: MagicMock() for f in indexed_fields}
        )
    return client


def _indexed(client) -> set[str]:
    return {c.kwargs["field_name"] for c in client.create_payload_index.call_args_list}


ALL_FIELDS = set(PAYLOAD_KEYWORD_FIELDS) | set(PAYLOAD_INTEGER_FIELDS)


class TestIndexesOnAnExistingCollection:
    def test_all_fields_are_indexed_when_none_exist(self):
        client = _client(collection_exists=True)
        QdrantStore(client, collection_name="existing_collection")
        assert _indexed(client) == ALL_FIELDS

    def test_the_collection_is_not_recreated(self):
        client = _client(collection_exists=True)
        QdrantStore(client, collection_name="existing_collection")
        client.create_collection.assert_not_called()

    def test_existing_indexes_are_left_alone(self):
        client = _client(collection_exists=True, indexed_fields=("cancer_type", "year"))
        QdrantStore(client, collection_name="existing_collection")
        created = _indexed(client)
        assert "cancer_type" not in created
        assert "year" not in created
        assert created == ALL_FIELDS - {"cancer_type", "year"}

    def test_nothing_is_created_when_all_exist(self):
        client = _client(collection_exists=True, indexed_fields=ALL_FIELDS)
        QdrantStore(client, collection_name="existing_collection")
        client.create_payload_index.assert_not_called()

    def test_cancer_type_is_covered(self):
        """Every /treatment-card request filters on it."""
        assert "cancer_type" in PAYLOAD_KEYWORD_FIELDS


class TestIndexingNeverBreaksStartup:
    """ensure_collection runs on every construction, including API startup."""

    def test_a_failing_index_call_does_not_raise(self):
        client = _client(collection_exists=True)
        client.create_payload_index.side_effect = RuntimeError("not permitted")
        QdrantStore(client, collection_name="existing_collection")  # must not raise

    def test_one_failure_does_not_stop_the_others(self):
        client = _client(collection_exists=True)
        calls = {"n": 0}

        def _sometimes_fail(**kwargs):
            calls["n"] += 1
            if kwargs["field_name"] == "section":
                raise RuntimeError("nope")

        client.create_payload_index.side_effect = _sometimes_fail
        QdrantStore(client, collection_name="existing_collection")
        assert calls["n"] == len(ALL_FIELDS)

    def test_unreadable_payload_schema_falls_back_to_creating_all(self):
        client = _client(collection_exists=True, get_collection_raises=True)
        QdrantStore(client, collection_name="existing_collection")
        assert _indexed(client) == ALL_FIELDS


class TestNewCollectionStillIndexed:
    def test_creating_a_collection_also_indexes_it(self):
        client = _client(collection_exists=False)
        QdrantStore(client, collection_name="existing_collection")
        client.create_collection.assert_called_once()
        assert _indexed(client) == ALL_FIELDS


class TestFilterFieldsAreAllIndexed:
    @pytest.mark.parametrize("field", sorted(ALL_FIELDS))
    def test_every_indexed_field_is_one_build_filter_uses(self, field):
        """Guards against indexing a field nothing filters on, and vice versa."""
        import inspect

        from src.db import vector_store

        src = inspect.getsource(vector_store._build_filter)
        probe = field if field not in ("year",) else "year_min"
        assert probe in src, f"{field} is indexed but _build_filter never uses it"


class TestCollectionNameIsRequired:
    """QdrantStore.collection_name used to default to a module constant reading
    "urological_oncology_v2", which names no collection that exists --
    production uses "urological_oncology_papers" from QDRANT_COLLECTION. Since
    ensure_collection creates a missing collection, anyone relying on that
    default silently got an empty one and served an empty corpus."""

    def test_omitting_it_is_a_type_error(self):
        with pytest.raises(TypeError):
            QdrantStore(_client(collection_exists=True))  # type: ignore[call-arg]

    def test_the_stale_constant_is_gone(self):
        from src.db import vector_store

        assert not hasattr(vector_store, "COLLECTION_NAME"), (
            "a module-level default collection name invites exactly the "
            "failure it used to cause"
        )

    def test_every_caller_passes_it_explicitly(self):
        """Guards the change: if a caller ever stops passing it, this and the
        TypeError test both fail rather than the mistake reaching production."""
        import re
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parents[2]
        offenders = []
        for path in list(root.glob("src/**/*.py")) + list(root.glob("scripts/**/*.py")):
            if "__pycache__" in str(path):
                continue
            for match in re.finditer(r"QdrantStore\(([^)]*)\)", path.read_text()):
                args = match.group(1)
                if args.strip() and "collection_name" not in args:
                    offenders.append(f"{path.name}: QdrantStore({args})")
        assert not offenders, offenders


class TestIndexRequestsAreAsynchronous:
    """The first production run of this code blocked on every index build. All
    six calls timed out after five seconds, adding 30 seconds to startup and
    logging six warnings saying the indexes did not exist -- when Qdrant had
    created every one of them."""

    def test_requests_do_not_wait_for_the_build(self):
        client = _client(collection_exists=True)
        QdrantStore(client, collection_name="existing_collection")
        for call in client.create_payload_index.call_args_list:
            assert call.kwargs.get("wait") is False, (
                "index creation must not block startup on building the index "
                "across every point"
            )

    def test_a_timeout_is_rechecked_before_being_reported_as_missing(self, caplog):
        """A client-side timeout is a wait expiring, not a rejection."""
        import logging

        client = _client(collection_exists=True)
        client.create_payload_index.side_effect = RuntimeError("timed out")
        # The re-read shows the indexes do exist: Qdrant accepted the requests.
        client.get_collection.side_effect = [
            MagicMock(payload_schema={}),                                  # before
            MagicMock(payload_schema={f: MagicMock() for f in ALL_FIELDS}),  # after
        ]
        with caplog.at_level(logging.INFO, logger="src.db.vector_store"):
            QdrantStore(client, collection_name="existing_collection")

        text = caplog.text
        assert "the index exists on" in text
        assert "will scan the collection" not in text, (
            "must not claim the field is unindexed without re-reading"
        )

    def test_a_genuine_failure_is_still_reported(self, caplog):
        import logging

        client = _client(collection_exists=True)
        client.create_payload_index.side_effect = RuntimeError("not permitted")
        client.get_collection.return_value = MagicMock(payload_schema={})
        with caplog.at_level(logging.WARNING, logger="src.db.vector_store"):
            QdrantStore(client, collection_name="existing_collection")
        assert "will scan the collection" in caplog.text
