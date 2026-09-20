"""
Integration tests for fallback visibility on /treatment-card.

When retrieval finds nothing, the endpoint generates the card from the model's
own clinical knowledge instead of refusing. That is intended behaviour and is
not changed here. The risk it carries is that such a card is structurally
identical to a grounded one -- same fields, same confidence wording, same
treatment table -- so a reader has no way to tell which they are holding.

Two signals now distinguish them:

  - retrieval_metadata["grounded"], reported unconditionally, because whether a
    card rests on retrieved literature is a property of the card and not of
    what the caller asked to be told
  - an explicit disclosure in `sources` instead of whatever citation the model
    produced from memory, on by default

The uro-rag-web frontend already sends disclose_fallback: true and already
renders the ungrounded state, so flipping the server default matches what the
live consumer was compensating for.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from starlette.testclient import TestClient

from src.retrieval.reranker import RankedChunk


def _ranked_chunk(chunk_id: str, score: float) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        text="Abiraterone improved overall survival in the treatment arm.",
        score=score,
        relevance_score=score,
        metadata={"pmid": "1", "pmcid": "1", "title": "T", "year": 2024,
                  "section": "results", "study_design": "rct",
                  "authors": ["Fizazi K"], "journal": "NEJM"},
    )

# ── Fallback visibility ──────────────────────────────────────────────────────

class TestUngroundedCardIsDistinguishable:
    """When retrieval finds nothing, /treatment-card generates from the model's
    own clinical knowledge. That is intended behaviour. The risk is not the
    fallback itself but that the resulting card is structurally identical to a
    grounded one: same fields, same confidence wording, same treatment table.

    These tests pin the two signals that tell them apart -- a machine-readable
    `grounded` flag that does not depend on any caller opt-in, and a
    human-readable disclosure in place of an empty source list."""

    def _card_response(self, chunks: list, body_extra: dict | None = None):
        from src.api.main import create_app
        from src.api.routes.treatment_card import (
            get_audit_logger, get_card_generator, get_retriever,
        )
        from src.generation.card_generator import CardGenerator
        from src.retrieval.retriever import RetrievalResult

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievalResult(
            query="q",
            chunks=chunks,
            retrieval_confidence=0.8 if chunks else 0.0,
            num_candidates=len(chunks),
            latency_ms={"total_ms": 10.0, "rerank_ms": 2.0},
        )

        llm = MagicMock()
        llm.model = "m"
        llm.provider = "anthropic"
        llm.complete.return_value = MagicMock(
            content="metastatic prostate cancer", input_tokens=1, output_tokens=1, model="m"
        )
        llm.complete_with_tools.return_value = {
            "input": {
                "stage": "mCRPC",
                "confidence": "Moderate",
                "guideline": "EAU 2024",
                "comorbidities_impact": "none",
                "treatment": [{"drug": "Abiraterone", "intent": "Palliative", "level": "A"}],
                "treatment_confidence": "Moderate",
                "sources": ["Fizazi et al. 2017 NEJM (RCT, n=1199)"],
            },
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }

        app = create_app()
        app.dependency_overrides[get_retriever] = lambda: mock_retriever
        app.dependency_overrides[get_card_generator] = lambda: CardGenerator(llm_client=llm)
        app.dependency_overrides[get_audit_logger] = lambda: None

        body = {
            "patient_id": "P1",
            "cancer_type": "prostate",
            "clinical_history": "mCRPC post-docetaxel, progressive disease",
            "language": "en",
        }
        body.update(body_extra or {})

        with TestClient(app, raise_server_exceptions=False) as client:
            return client.post("/treatment-card", json=body, headers={"X-API-Key": "dev"})

    def test_grounded_false_when_nothing_retrieved(self):
        resp = self._card_response([])
        assert resp.status_code == 200
        assert resp.json()["retrieval_metadata"]["grounded"] is False

    def test_grounded_true_when_chunks_retrieved(self):
        resp = self._card_response([_ranked_chunk("c1", 0.8)])
        assert resp.status_code == 200
        assert resp.json()["retrieval_metadata"]["grounded"] is True

    def test_grounded_is_reported_even_when_disclosure_is_switched_off(self):
        """The flag is a property of the card, not of the caller's opt-in."""
        resp = self._card_response([], {"disclose_fallback": False})
        assert resp.json()["retrieval_metadata"]["grounded"] is False

    def test_disclosure_replaces_sources_by_default(self):
        """Without opting in to anything, an ungrounded card must not present
        the model's remembered citation as though it were retrieved."""
        resp = self._card_response([])
        sources = resp.json()["sources"]
        assert len(sources) == 1
        assert "No relevant literature" in sources[0]
        assert "Fizazi" not in sources[0]

    def test_sources_detail_marks_the_entry_as_parametric(self):
        detail = self._card_response([]).json()["retrieval_metadata"]["sources_detail"]
        assert len(detail) == 1
        assert detail[0]["study_design"] == "parametric_knowledge"
        assert detail[0]["pmid"] == ""

    def test_grounded_card_keeps_its_real_sources(self):
        resp = self._card_response([_ranked_chunk("c1", 0.8)])
        assert "No relevant literature" not in " ".join(resp.json()["sources"])
