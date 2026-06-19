"""
Unit tests for the cancer_type/cancer_types Pydantic validators in
src/api/routes/query.py and src/api/routes/treatment_card.py.

Regression coverage: whitespace around a valid topic name (e.g. "prostate ")
must not be rejected — the validators must strip before normalising.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.routes.query import QueryRequest
from src.api.routes.treatment_card import TreatmentCardRequest

_VALID_CARD_KWARGS = dict(
    patient_id="P1",
    clinical_history="Metastatic prostate cancer, PSA 42 ng/mL.",
)


class TestQueryRequestCancerTypes:
    def test_trailing_space_is_accepted(self):
        req = QueryRequest(query="q", cancer_types=["prostate "])
        assert req.cancer_types == ["prostate"]

    def test_leading_space_is_accepted(self):
        req = QueryRequest(query="q", cancer_types=[" bladder"])
        assert req.cancer_types == ["bladder"]

    def test_alias_with_whitespace_resolves(self):
        req = QueryRequest(query="q", cancer_types=[" renal "])
        assert req.cancer_types == ["kidney"]

    def test_unsupported_topic_still_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="q", cancer_types=["lung"])


class TestTreatmentCardRequestCancerType:
    def test_trailing_space_is_accepted(self):
        req = TreatmentCardRequest(cancer_type="prostate ", **_VALID_CARD_KWARGS)
        assert req.cancer_type == "prostate"

    def test_leading_space_is_accepted(self):
        req = TreatmentCardRequest(cancer_type=" bladder", **_VALID_CARD_KWARGS)
        assert req.cancer_type == "bladder"

    def test_alias_with_whitespace_resolves(self):
        req = TreatmentCardRequest(cancer_type=" rcc ", **_VALID_CARD_KWARGS)
        assert req.cancer_type == "kidney"

    def test_unsupported_topic_still_rejected(self):
        with pytest.raises(ValidationError):
            TreatmentCardRequest(cancer_type="lung", **_VALID_CARD_KWARGS)
