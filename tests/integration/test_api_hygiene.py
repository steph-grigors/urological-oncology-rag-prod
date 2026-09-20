"""
Integration tests for the API hardening in fix/api-observability-hygiene.

Three behaviours, each of which was previously wrong in a way no test caught:

  1. /ingestion/status was reachable with no credentials at all.
  2. The health probes, which are deliberately unauthenticated, returned the
     raw driver exception string. Those routinely carry the host, port,
     database name or credentials from a DSN.
  3. /query's latency breakdown summed overlapping spans, so `retrieval`
     included reranking and the total again.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from starlette.testclient import TestClient

from src.generation.generator import GenerationResult
from src.retrieval.reranker import RankedChunk
from src.retrieval.retriever import RetrievalResult


def _ranked_chunk(chunk_id: str, score: float) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        text="Enzalutamide improved overall survival in the treatment arm.",
        score=score,
        relevance_score=score,
        metadata={"pmid": "1", "pmcid": "1", "title": "T", "year": 2024,
                  "section": "results", "study_design": "rct"},
    )


# ── /ingestion/status ────────────────────────────────────────────────────────

class TestIngestionStatusRequiresAuth:
    """The endpoint exposes corpus size, per-topic counts and spend. It is not
    a health probe and must not be readable anonymously."""

    def _app(self):
        from src.api.main import create_app
        from config.settings import get_settings, Settings

        app = create_app()
        settings = Settings(
            OPENAI_API_KEY="x",
            APP_ENV="production",          # no development auth bypass
            API_KEYS=["valid-key"],
        )
        app.dependency_overrides[get_settings] = lambda: settings
        return app

    def test_rejects_request_with_no_key(self):
        with TestClient(self._app(), raise_server_exceptions=False) as client:
            assert client.get("/ingestion/status").status_code == 401

    def test_rejects_request_with_wrong_key(self):
        with TestClient(self._app(), raise_server_exceptions=False) as client:
            resp = client.get("/ingestion/status", headers={"X-API-Key": "nope"})
            assert resp.status_code == 401

    def test_accepts_a_valid_key(self):
        """A valid key gets past auth. The response is then either the progress
        file or a 404 saying there is no run, depending on whether the file
        exists in the working directory -- both mean authentication passed."""
        with TestClient(self._app(), raise_server_exceptions=False) as client:
            resp = client.get("/ingestion/status", headers={"X-API-Key": "valid-key"})
            assert resp.status_code != 401


# ── Health probes ────────────────────────────────────────────────────────────

class TestHealthProbesDoNotLeakExceptionDetail:
    """These two routes are intentionally unauthenticated for load balancers,
    so whatever they return is public."""

    # Anything assigned to app.state before startup is discarded: the lifespan
    # handler reassigns retriever/audit_logger as it tries to build them. So
    # the failing doubles have to be installed after the client has started.
    LEAKY_QDRANT = "could not connect to host=10.0.0.7 port=5433 user=rag password=hunter2"
    LEAKY_PG = 'FATAL: password authentication failed for user "rag"'
    SECRETS = ("hunter2", "10.0.0.7", "5433", "user=rag", "password authentication")

    @staticmethod
    def _install_failing_backends(app):
        boom = MagicMock()
        boom._store.collection_stats.side_effect = RuntimeError(
            TestHealthProbesDoNotLeakExceptionDetail.LEAKY_QDRANT
        )
        app.state.retriever = boom

        bad_db = MagicMock()
        bad_db.engine.connect.side_effect = RuntimeError(
            TestHealthProbesDoNotLeakExceptionDetail.LEAKY_PG
        )
        app.state.audit_logger = bad_db

    def _probe(self, path: str):
        from src.api.main import create_app

        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            self._install_failing_backends(app)
            return client.get(path)

    def test_health_reports_failure_without_the_message(self):
        resp = self._probe("/health")
        assert resp.status_code == 503
        assert resp.json()["checks"]["qdrant"] == "error"
        assert resp.json()["checks"]["postgres"] == "error"
        for secret in self.SECRETS:
            assert secret not in resp.text, f"{secret!r} leaked into /health"

    def test_ready_reports_failure_without_the_message(self):
        resp = self._probe("/health/ready")
        assert resp.status_code == 503
        assert resp.json()["checks"]["qdrant"] == "error"
        assert resp.json()["checks"]["postgres"] == "error"
        for secret in self.SECRETS:
            assert secret not in resp.text, f"{secret!r} leaked into /health/ready"


# ── /query latency breakdown ─────────────────────────────────────────────────

class TestLatencyBreakdownDoesNotDoubleCount:
    def test_retrieval_excludes_rerank_and_total(self):
        """`total_ms` from the retriever already covers every retrieval phase
        including rerank. Summing the dict added each phase twice and the
        total on top: 10 + 5 + 8 + 23 = 46 for a 23 ms retrieval."""
        from src.api.main import create_app
        from src.api.routes.query import (
            get_audit_logger, get_generator, get_retriever,
        )

        chunks = [_ranked_chunk("c1", 0.8)]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievalResult(
            query="q",
            chunks=chunks,
            retrieval_confidence=0.8,
            num_candidates=1,
            latency_ms={"embed_ms": 4.0, "dense_ms": 6.0, "bm25_ms": 5.0,
                        "rerank_ms": 8.0, "total_ms": 23.0},
        )

        mock_generator = MagicMock()
        mock_generator.generate.return_value = GenerationResult(
            answer="Enzalutamide improves survival [Doc 1].",
            citations=[1],
            evidence_quality="high",
            model_used="m",
            provider="anthropic",
            prompt_tokens=10,
            completion_tokens=5,
            confidence_score=0.8,
            hallucinated_citations=[],
            latency_ms=12.0,
        )

        app = create_app()
        app.dependency_overrides[get_retriever] = lambda: mock_retriever
        app.dependency_overrides[get_generator] = lambda: mock_generator
        app.dependency_overrides[get_audit_logger] = lambda: None

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/query",
                json={"query": "What is the efficacy of enzalutamide?"},
                headers={"X-API-Key": "dev"},
            )

        assert resp.status_code == 200
        latency = resp.json()["latency_ms"]

        assert latency["rerank"] == 8
        assert latency["retrieval"] == 15          # 23 total - 8 rerank
        assert latency["retrieval"] != 46          # the old double-counted value
        assert latency["retrieval"] + latency["rerank"] == 23
