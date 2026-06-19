"""
Unit tests for AuditLogger.log_treatment_card (src/observability/audit.py).

Uses an in-memory SQLite engine — no external Postgres required.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from src.db.models import AuditLog
from src.generation.card_generator import TreatmentCardResult, TreatmentTriplet
from src.observability.audit import AuditLogger


@pytest.fixture
def audit_logger(tmp_path):
    # File-backed (not ":memory:") so the same DB persists across the pooled
    # connections SQLAlchemy opens for create_all() vs. the actual insert.
    db_path = tmp_path / f"audit_{uuid.uuid4().hex}.db"
    return AuditLogger(f"sqlite:///{db_path}")


def _make_card_result(**overrides) -> TreatmentCardResult:
    defaults = dict(
        patient_id="P1",
        stage="cT3b N1 M1b",
        confidence="High",
        guideline="EAU 2024",
        comorbidities_impact="None.",
        treatment=[TreatmentTriplet(drug="ADT", intent="Palliative", level="A")],
        treatment_confidence="High",
        sources=["Fizazi et al. 2017 NEJM (RCT, n=1199)"],
        retrieval_metadata={
            "chunks_used": 1,
            "confidence_score": 0.82,
            "corpus_version": "",
            "sources_detail": [{"title": "Fizazi RCT", "year": 2017, "pmid": "28593635"}],
        },
        prompt_tokens=100,
        completion_tokens=200,
        latency_ms=2500.0,
    )
    defaults.update(overrides)
    return TreatmentCardResult(**defaults)


class TestLogTreatmentCard:
    def test_writes_one_row(self, audit_logger):
        logger = audit_logger
        asyncio.run(
            logger.log_treatment_card(
                query_id="q1",
                patient_id="P1",
                clinical_history="mCRPC, PSA 42",
                card_result=_make_card_result(),
                model="claude-sonnet-4-6",
                provider="anthropic",
                user_id="key123",
                session_id="conv1",
            )
        )
        from sqlalchemy.orm import Session
        with Session(logger.engine) as session:
            rows = session.query(AuditLog).all()
        assert len(rows) == 1

    def test_row_contains_grounded_sources_and_card_json(self, audit_logger):
        logger = audit_logger
        asyncio.run(
            logger.log_treatment_card(
                query_id="q2",
                patient_id="P2",
                clinical_history="mCRPC",
                card_result=_make_card_result(patient_id="P2"),
                model="claude-sonnet-4-6",
                provider="anthropic",
            )
        )
        from sqlalchemy.orm import Session
        with Session(logger.engine) as session:
            row = session.query(AuditLog).filter_by(query_id="q2").one()

        assert row.model == "claude-sonnet-4-6"
        assert row.provider == "anthropic"
        assert row.confidence == 0.82
        assert row.input_tokens == 100
        assert row.output_tokens == 200
        assert row.sources[0]["title"] == "Fizazi RCT"
        card = json.loads(row.answer)
        assert card["stage"] == "cT3b N1 M1b"
        assert card["treatment"][0]["drug"] == "ADT"

    def test_ungrounded_card_keeps_sources_column_empty_not_llm_text(self, audit_logger):
        """Regression test: zero chunks + disclose_fallback=False (no 'grounded' key,
        sources_detail=[]) must NOT cause the LLM's free-text `sources` to be
        silently stored in the `sources` column disguised as real citations."""
        logger = audit_logger
        card = _make_card_result(
            patient_id="P3",
            retrieval_metadata={"chunks_used": 0, "confidence_score": 0.1, "corpus_version": ""},
        )
        asyncio.run(
            logger.log_treatment_card(
                query_id="q3",
                patient_id="P3",
                clinical_history="insufficient evidence case",
                card_result=card,
                model="claude-sonnet-4-6",
                provider="anthropic",
            )
        )
        from sqlalchemy.orm import Session
        with Session(logger.engine) as session:
            row = session.query(AuditLog).filter_by(query_id="q3").one()
        assert row.sources == []  # never silently substituted with LLM text
        assert row.flagged is True  # ungrounded card — flagged for review

    def test_answer_json_records_chunks_used_and_grounded(self, audit_logger):
        """retrieval_metadata.chunks_used/grounded must be queryable from the
        stored row regardless of whether the API caller opted into
        disclose_fallback — auditing isn't gated by the API response contract."""
        logger = audit_logger
        asyncio.run(
            logger.log_treatment_card(
                query_id="q4",
                patient_id="P4",
                clinical_history="mCRPC",
                card_result=_make_card_result(patient_id="P4"),
                model="claude-sonnet-4-6",
                provider="anthropic",
            )
        )
        from sqlalchemy.orm import Session
        with Session(logger.engine) as session:
            row = session.query(AuditLog).filter_by(query_id="q4").one()
        card = json.loads(row.answer)
        assert card["retrieval_metadata"]["chunks_used"] == 1
        assert card["retrieval_metadata"]["grounded"] is True
        assert row.flagged is False

    def test_grounded_derived_from_chunks_used_when_flag_absent(self, audit_logger):
        """When disclose_fallback wasn't set, retrieval_metadata has no 'grounded'
        key — the audit log must still derive it from chunks_used rather than
        assuming grounded=True by default."""
        logger = audit_logger
        card = _make_card_result(
            patient_id="P5",
            retrieval_metadata={"chunks_used": 0, "confidence_score": 0.05, "corpus_version": ""},
        )
        asyncio.run(
            logger.log_treatment_card(
                query_id="q5",
                patient_id="P5",
                clinical_history="no evidence found",
                card_result=card,
                model="claude-sonnet-4-6",
                provider="anthropic",
            )
        )
        from sqlalchemy.orm import Session
        with Session(logger.engine) as session:
            row = session.query(AuditLog).filter_by(query_id="q5").one()
        card_json = json.loads(row.answer)
        assert card_json["retrieval_metadata"]["grounded"] is False
        assert row.flagged is True
