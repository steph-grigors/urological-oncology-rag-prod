"""
Route-level behaviour of the per-request model picker.

Covers the three things that must hold for the picker to be safe to expose:
an off-allowlist model is refused, omitting the field preserves the previous
behaviour exactly, and a selected model actually changes who answers.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.generation.card_generator import TreatmentCardResult, TreatmentTriplet
from src.retrieval.retriever import RetrievalResult

_CARD_REQUEST = {
    "patient_id": "P1",
    "cancer_type": "prostate",
    "clinical_history": "Patient avec cancer de prostate metastatique hormono-naif cT3b N1 M1b.",
    "comorbidities": {},
    "top_k": 3,
}


def _retrieval_result() -> RetrievalResult:
    chunk = MagicMock()
    chunk.text = "ADT + abiraterone improved OS in mHSPC."
    chunk.metadata = {"title": "LATITUDE", "year": 2017, "study_design": "rct", "sample_size": 1199}
    chunk.relevance_score = 0.85
    return RetrievalResult(
        query="metastatic prostate cancer",
        chunks=[chunk],
        retrieval_confidence=0.85,
        num_candidates=10,
        latency_ms={"dense_ms": 50, "bm25_ms": 10, "rerank_ms": 80},
    )


def _card_result() -> TreatmentCardResult:
    return TreatmentCardResult(
        patient_id="P1",
        stage="cT3b N1 M1b",
        confidence="High",
        guideline="EAU 2024",
        comorbidities_impact="None.",
        treatment=[TreatmentTriplet(drug="ADT", intent="Palliative", level="A", warnings=[])],
        treatment_confidence="High",
        sources=["Fizazi et al. 2017 NEJM"],
        retrieval_metadata={"chunks_used": 1, "confidence_score": 0.85, "corpus_version": ""},
        prompt_tokens=150,
        completion_tokens=300,
        latency_ms=2500.0,
    )


def _card_gen(model: str = "claude-sonnet-4-6") -> MagicMock:
    gen = MagicMock()
    gen.model = model
    gen.provider = "anthropic"
    gen.translate_to_english.return_value = "metastatic prostate cancer"
    gen.generate_card.return_value = _card_result()
    return gen


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.retriever = MagicMock(**{"retrieve.return_value": _retrieval_result()})
        app.state.card_generator = _card_gen()
        yield c


# ── Allowlist enforcement ─────────────────────────────────────────────────────

@pytest.mark.parametrize("bogus", ["gpt-4o", "claude-fable-5-1", "claude-3-opus-20240229"])
def test_card_rejects_off_allowlist_model(client, bogus):
    resp = client.post(
        "/treatment-card",
        json={**_CARD_REQUEST, "model": bogus},
        headers={"X-API-Key": "dev"},
    )
    assert resp.status_code == 400
    assert "not selectable" in resp.json()["detail"]


def test_rejection_happens_before_any_generation(client):
    """A refused model must not burn a retrieval or an LLM call."""
    gen = client.app.state.card_generator
    client.post(
        "/treatment-card",
        json={**_CARD_REQUEST, "model": "gpt-4o"},
        headers={"X-API-Key": "dev"},
    )
    gen.generate_card.assert_not_called()
    gen.translate_to_english.assert_not_called()


# ── Default path is unchanged ─────────────────────────────────────────────────

def test_card_without_model_field_uses_default_instance(client):
    """Omitting `model` must route to the exact object wired at startup."""
    gen = client.app.state.card_generator
    resp = client.post("/treatment-card", json=_CARD_REQUEST, headers={"X-API-Key": "dev"})
    assert resp.status_code == 200
    gen.generate_card.assert_called_once()
    assert resp.json()["model_used"] == "claude-sonnet-4-6"


def test_naming_the_default_model_reuses_the_default_instance(client):
    """Explicitly asking for the default must not build a second generator."""
    gen = client.app.state.card_generator
    resp = client.post(
        "/treatment-card",
        json={**_CARD_REQUEST, "model": "claude-sonnet-4-6"},
        headers={"X-API-Key": "dev"},
    )
    assert resp.status_code == 200
    gen.generate_card.assert_called_once()


# ── Selection actually takes effect ───────────────────────────────────────────

def test_selecting_another_model_bypasses_the_default_instance(client, monkeypatch):
    built = {}

    def fake_build(llm_client):
        built["model"] = llm_client.model
        return _card_gen(llm_client.model)

    import src.api.model_selection as sel
    monkeypatch.setattr(sel, "_client_for", lambda app, mid: MagicMock(model=mid, provider="anthropic"))
    monkeypatch.setattr(
        sel, "card_generator_for",
        lambda request, requested, default: sel._select(request, requested, default, fake_build),
    )
    import src.api.routes.treatment_card as tc
    monkeypatch.setattr(tc, "card_generator_for", sel.card_generator_for)

    default_gen = client.app.state.card_generator
    resp = client.post(
        "/treatment-card",
        json={**_CARD_REQUEST, "model": "claude-opus-5"},
        headers={"X-API-Key": "dev"},
    )
    assert resp.status_code == 200
    assert built["model"] == "claude-opus-5"
    assert resp.json()["model_used"] == "claude-opus-5"
    default_gen.generate_card.assert_not_called()


# ── Discovery endpoint ────────────────────────────────────────────────────────

def test_models_endpoint_lists_the_allowlist(client):
    body = client.get("/models").json()
    assert [m["id"] for m in body["models"]] == [
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        "claude-opus-5",
    ]
    assert body["default"] == "claude-sonnet-4-6"
    assert sum(m["is_default"] for m in body["models"]) == 1
