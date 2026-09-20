"""
Integration tests for the readiness probe's empty-collection check.

QdrantStore.ensure_collection creates the collection when it is missing. A typo
in QDRANT_COLLECTION therefore does not fail anything: a brand-new empty
collection appears, collection_stats() returns cleanly, and the probe reports
"ok" while the API serves an empty corpus. Retrieval finds nothing, every query
falls through to the parametric-knowledge path, and nothing says why.

Reporting reachable-but-empty as a failure is what turns that silent
misconfiguration into a loud one.

Neither container healthcheck is affected: docker/Dockerfile and
docker/docker-compose.yml both probe /health/live, which is untouched.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient


def _app_with_collection(point_count, *, raises: bool = False):
    from src.api.main import create_app

    app = create_app()
    retriever = MagicMock()
    if raises:
        retriever._store.collection_stats.side_effect = RuntimeError("qdrant unreachable")
    else:
        retriever._store.collection_stats.return_value = {
            "collection": "urological_oncology_papers",
            "point_count": point_count,
            "status": "green",
            "dense_vector_size": 1536,
        }

    audit_logger = MagicMock()
    audit_logger.engine.connect.return_value.__enter__ = lambda s: MagicMock()
    audit_logger.engine.connect.return_value.__exit__ = lambda s, *a: None
    return app, retriever, audit_logger


def _probe(path: str, point_count, *, raises: bool = False):
    app, retriever, audit_logger = _app_with_collection(point_count, raises=raises)
    with TestClient(app, raise_server_exceptions=False) as client:
        # Installed after startup: the lifespan reassigns app.state as it tries
        # to build the real stack.
        app.state.retriever = retriever
        app.state.audit_logger = audit_logger
        return client.get(path)


@pytest.mark.parametrize("path", ["/health", "/health/ready"])
class TestEmptyCollectionIsNotHealthy:
    def test_populated_collection_is_ok(self, path):
        resp = _probe(path, 795_306)
        assert resp.status_code == 200
        assert resp.json()["checks"]["qdrant"] == "ok"
        assert resp.json()["status"] == "ok"

    def test_zero_points_is_reported_as_empty(self, path):
        resp = _probe(path, 0)
        assert resp.status_code == 503
        assert resp.json()["checks"]["qdrant"] == "empty"
        assert resp.json()["status"] == "degraded"

    def test_missing_point_count_is_reported_as_empty(self, path):
        """A stats payload without the key is treated the same way rather than
        being read as healthy."""
        resp = _probe(path, None)
        assert resp.status_code == 503
        assert resp.json()["checks"]["qdrant"] == "empty"

    def test_unreachable_qdrant_is_still_an_error_not_empty(self, path):
        resp = _probe(path, 0, raises=True)
        assert resp.status_code == 503
        assert resp.json()["checks"]["qdrant"] == "error"

    def test_empty_does_not_leak_the_collection_name(self, path):
        """Same reasoning as the exception-detail fix: these probes are
        unauthenticated, so the diagnosis goes to the log."""
        resp = _probe(path, 0)
        assert "urological_oncology_papers" not in resp.text


class TestLivenessIsUnaffected:
    """The container healthcheck probes /health/live. It must keep returning
    200 on an empty collection, or compose would restart-loop a container whose
    only problem is that ingestion has not run yet."""

    def test_live_is_200_with_an_empty_collection(self):
        resp = _probe("/health/live", 0)
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_live_is_200_when_qdrant_is_unreachable(self):
        resp = _probe("/health/live", 0, raises=True)
        assert resp.status_code == 200
