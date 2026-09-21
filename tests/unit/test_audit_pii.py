"""
Tests for de-identification on the way into the audit log.

config.constants.PII_SCRUB_PATTERNS described what should be stripped from
persisted free text and was referenced from nowhere, so every question and
every patient narrative was written verbatim. Confirmed in production on
2026-09-21: 86 rows holding a patient identifier and clinical history, in a
database that was reachable from the open internet.
"""

from __future__ import annotations

import inspect

import pytest

from src.observability import audit
from src.observability.scrub import (
    audit_question_for_card,
    pseudonymise_patient_id,
    scrub_pii,
)


class TestPatientIdPseudonymisation:
    def test_token_does_not_contain_the_identifier(self):
        token = pseudonymise_patient_id("P-12345")
        assert "12345" not in token
        assert token.startswith("p_")

    def test_same_patient_maps_to_the_same_token(self):
        """A reviewer must still be able to see that several cards concern one
        patient, and count distinct patients."""
        assert pseudonymise_patient_id("P-1") == pseudonymise_patient_id("P-1")

    def test_different_patients_map_to_different_tokens(self):
        assert pseudonymise_patient_id("P-1") != pseudonymise_patient_id("P-2")

    def test_empty_stays_empty(self):
        """A caller that sent no identifier must not get a token implying one."""
        assert pseudonymise_patient_id("") == ""

    def test_token_fits_the_column(self):
        assert len(pseudonymise_patient_id("x" * 500)) <= 100


class TestPiiScrubbing:
    @pytest.mark.parametrize(
        "raw,leaked",
        [
            ("SSN 123-45-6789 noted", "123-45-6789"),
            ("MRN 1234567890123 on file", "1234567890123"),
            ("contact oncologist@hospital.org", "oncologist@hospital.org"),
        ],
    )
    def test_structured_identifiers_are_removed(self, raw, leaked):
        assert leaked not in scrub_pii(raw)
        assert "[redacted]" in scrub_pii(raw)

    def test_clinical_content_is_preserved(self):
        """Scrubbing must not gut the text -- the row exists to record what the
        system was asked."""
        text = "mCRPC post-docetaxel, PSA 42, ECOG 1, progressive bone disease"
        assert scrub_pii(text) == text

    def test_empty_input_is_safe(self):
        assert scrub_pii("") == ""


class TestCardQuestionShape:
    def test_shape_is_unchanged_so_existing_queries_still_match(self):
        q = audit_question_for_card("P-77", "mCRPC post-docetaxel")
        assert q.startswith("[treatment-card] patient=")
        assert "treatment-card" in q

    def test_identifier_is_pseudonymised_in_the_stored_string(self):
        q = audit_question_for_card("P-77", "mCRPC post-docetaxel")
        assert "patient=P-77" not in q
        assert "patient=p_" in q

    def test_narrative_is_scrubbed_in_the_stored_string(self):
        q = audit_question_for_card("P-77", "mCRPC, MRN 9988776655443")
        assert "9988776655443" not in q

    def test_narrative_is_still_truncated(self):
        q = audit_question_for_card("P-1", "x" * 5000)
        assert len(q) < 700


class TestBothAuditPathsUseIt:
    def test_treatment_card_path_pseudonymises(self):
        src = inspect.getsource(audit.AuditLogger.log_treatment_card)
        assert "audit_question_for_card(" in src
        assert "patient={patient_id}" not in src

    def test_query_path_scrubs(self):
        """A clinician can paste an identifier into a free-text question."""
        src = inspect.getsource(audit.AuditLogger.log)
        assert "scrub_pii(question)" in src

    def test_card_answer_carries_no_patient_identifier(self):
        """The card JSON never included one; this pins that it stays that way."""
        src = inspect.getsource(audit.AuditLogger.log_treatment_card)
        card_json = src[src.index("card_json = json.dumps"):src.index("await asyncio.to_thread")]
        assert "patient_id" not in card_json
