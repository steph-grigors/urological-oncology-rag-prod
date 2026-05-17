"""
Integration tests for POST /treatment-card.

Uses FastAPI TestClient with mocked retriever + CardGenerator so no real
LLM or Qdrant calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.generation.card_generator import (
    TreatmentCardResult,
    TreatmentTriplet,
    TreatmentWarning,
)
from src.retrieval.retriever import RetrievalResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

_VALID_REQUEST = {
    "patient_id": "P1",
    "cancer_type": "prostate",
    "age_range": "70–79 ans",
    "clinical_history": (
        "Patient avec cancer de prostate métastatique hormono-naïf cT3b N1 M1b, "
        "ISUP 4, PSA 42 ng/mL. Pas d'antécédent de traitement."
    ),
    "comorbidities": {"Insuffisance rénale chronique": "stade 3"},
    "top_k": 3,
}


def _make_retrieval_result() -> RetrievalResult:
    chunk = MagicMock()
    chunk.text = "ADT + abiraterone improved OS in mHSPC."
    chunk.metadata = {"title": "LATITUDE", "year": 2017, "study_design": "rct", "sample_size": 1199}
    chunk.relevance_score = 0.85
    return RetrievalResult(
        query="metastatic prostate cancer first line",
        chunks=[chunk],
        retrieval_confidence=0.85,
        num_candidates=10,
        latency_ms={"dense_ms": 50, "bm25_ms": 10, "rerank_ms": 80},
    )


def _make_card_result() -> TreatmentCardResult:
    return TreatmentCardResult(
        patient_id="P1",
        stage="cT3b N1 M1b, ISUP 4, PSA 42 ng/mL",
        confidence="High",
        guideline="EAU 2024",
        comorbidities_impact="Moderate renal impairment — adjust dosage.",
        treatment=[
            TreatmentTriplet(
                drug="ADT + Abiraterone 1000 mg/day + Prednisone 10 mg/day",
                intent="Palliative",
                level="A",
                warnings=[],
            ),
        ],
        treatment_confidence="High",
        sources=["Fizazi et al. 2017 NEJM (RCT phase III, n=1199)"],
        retrieval_metadata={"chunks_used": 1, "confidence_score": 0.85, "corpus_version": ""},
        prompt_tokens=150,
        completion_tokens=300,
        latency_ms=2500.0,
    )


@pytest.fixture
def client():
    app = create_app()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = _make_retrieval_result()

    mock_card_gen = MagicMock()
    mock_card_gen.translate_to_english.return_value = "metastatic prostate cancer first line"
    mock_card_gen.generate_card.return_value = _make_card_result()

    with TestClient(app, raise_server_exceptions=True) as c:
        # Inject mocks AFTER lifespan runs (lifespan overwrites state with None
        # when Qdrant/Postgres are unavailable in the test environment)
        app.state.retriever = mock_retriever
        app.state.card_generator = mock_card_gen
        yield c


# ── Happy path ────────────────────────────────────────────────────────────────

class TestTreatmentCardHappyPath:
    def test_returns_200(self, client):
        resp = client.post(
            "/treatment-card",
            json=_VALID_REQUEST,
            headers={"X-API-Key": "dev"},
        )
        assert resp.status_code == 200

    def test_patient_id_echoed(self, client):
        resp = client.post("/treatment-card", json=_VALID_REQUEST, headers={"X-API-Key": "dev"})
        assert resp.json()["patient_id"] == "P1"

    def test_stage_present(self, client):
        resp = client.post("/treatment-card", json=_VALID_REQUEST, headers={"X-API-Key": "dev"})
        assert "PSA" in resp.json()["stage"]

    def test_treatment_array_returned(self, client):
        resp = client.post("/treatment-card", json=_VALID_REQUEST, headers={"X-API-Key": "dev"})
        treatment = resp.json()["treatment"]
        assert isinstance(treatment, list)
        assert len(treatment) == 1
        assert treatment[0]["drug"] == "ADT + Abiraterone 1000 mg/day + Prednisone 10 mg/day"
        assert treatment[0]["intent"] == "Palliative"
        assert treatment[0]["level"] == "A"

    def test_warnings_field_present_on_triplets(self, client):
        resp = client.post("/treatment-card", json=_VALID_REQUEST, headers={"X-API-Key": "dev"})
        assert "warnings" in resp.json()["treatment"][0]

    def test_sources_returned(self, client):
        resp = client.post("/treatment-card", json=_VALID_REQUEST, headers={"X-API-Key": "dev"})
        sources = resp.json()["sources"]
        assert isinstance(sources, list)
        assert len(sources) > 0
        assert "Fizazi" in sources[0]

    def test_retrieval_metadata_present(self, client):
        resp = client.post("/treatment-card", json=_VALID_REQUEST, headers={"X-API-Key": "dev"})
        meta = resp.json()["retrieval_metadata"]
        assert meta["chunks_used"] == 1
        assert meta["confidence_score"] == 0.85

    def test_request_id_in_response(self, client):
        resp = client.post("/treatment-card", json=_VALID_REQUEST, headers={"X-API-Key": "dev"})
        assert "request_id" in resp.json()

    def test_latency_ms_present(self, client):
        resp = client.post("/treatment-card", json=_VALID_REQUEST, headers={"X-API-Key": "dev"})
        assert isinstance(resp.json()["latency_ms"], int)


# ── Validation errors ──────────────────────────────────────────────────────────

class TestTreatmentCardValidation:
    def test_missing_patient_id_returns_422(self, client):
        body = dict(_VALID_REQUEST)
        del body["patient_id"]
        resp = client.post("/treatment-card", json=body, headers={"X-API-Key": "dev"})
        assert resp.status_code == 422

    def test_missing_cancer_type_returns_422(self, client):
        body = dict(_VALID_REQUEST)
        del body["cancer_type"]
        resp = client.post("/treatment-card", json=body, headers={"X-API-Key": "dev"})
        assert resp.status_code == 422

    def test_short_clinical_history_returns_422(self, client):
        body = dict(_VALID_REQUEST, clinical_history="short")
        resp = client.post("/treatment-card", json=body, headers={"X-API-Key": "dev"})
        assert resp.status_code == 422

    def test_top_k_above_max_returns_422(self, client):
        body = dict(_VALID_REQUEST, top_k=20)
        resp = client.post("/treatment-card", json=body, headers={"X-API-Key": "dev"})
        assert resp.status_code == 422


# ── Service unavailable ────────────────────────────────────────────────────────

class TestTreatmentCardServiceUnavailable:
    def test_503_when_retriever_none(self):
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.retriever = None
            app.state.card_generator = MagicMock()
            resp = c.post("/treatment-card", json=_VALID_REQUEST, headers={"X-API-Key": "dev"})
        assert resp.status_code == 503

    def test_503_when_card_generator_none(self):
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.retriever = MagicMock()
            app.state.card_generator = None
            resp = c.post("/treatment-card", json=_VALID_REQUEST, headers={"X-API-Key": "dev"})
        assert resp.status_code == 503


# ── Warnings propagated through route ─────────────────────────────────────────

class TestWarningsPropagation:
    def test_regulatory_warning_in_response(self):
        app = create_app()
        card_with_warning = _make_card_result()
        card_with_warning.treatment[0].warnings = [
            TreatmentWarning(
                type="regulatory",
                drug="atezolizumab",
                jurisdiction="EMA",
                message="Atézolizumab — AMM EMA retirée (2021).",
            )
        ]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_retrieval_result()
        mock_card_gen = MagicMock()
        mock_card_gen.translate_to_english.return_value = "bladder cancer"
        mock_card_gen.generate_card.return_value = card_with_warning

        with TestClient(app, raise_server_exceptions=True) as c:
            app.state.retriever = mock_retriever
            app.state.card_generator = mock_card_gen
            resp = c.post("/treatment-card", json=_VALID_REQUEST, headers={"X-API-Key": "dev"})

        warnings = resp.json()["treatment"][0]["warnings"]
        assert len(warnings) == 1
        assert warnings[0]["type"] == "regulatory"
        assert "AMM EMA retirée" in warnings[0]["message"]

    def test_biomarker_warning_in_response(self):
        app = create_app()
        card_with_warning = _make_card_result()
        card_with_warning.treatment[0].warnings = [
            TreatmentWarning(
                type="biomarker",
                drug="lu-177-psma-617",
                jurisdiction="biomarker",
                message="Lu-177 requiert une positivité PSMA-PET confirmée.",
            )
        ]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_retrieval_result()
        mock_card_gen = MagicMock()
        mock_card_gen.translate_to_english.return_value = "prostate cancer"
        mock_card_gen.generate_card.return_value = card_with_warning

        with TestClient(app, raise_server_exceptions=True) as c:
            app.state.retriever = mock_retriever
            app.state.card_generator = mock_card_gen
            resp = c.post("/treatment-card", json=_VALID_REQUEST, headers={"X-API-Key": "dev"})

        warnings = resp.json()["treatment"][0]["warnings"]
        assert warnings[0]["type"] == "biomarker"
        assert "PSMA-PET" in warnings[0]["message"]
