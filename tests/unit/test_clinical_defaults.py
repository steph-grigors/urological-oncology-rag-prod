"""
Tests for the points where the card pipeline used to invent a clinical signal.

Four separate places filled a gap with a confident-looking value rather than an
honest one. Each is small; each puts a specific wrong claim on a card an intern
may act on.
"""

from __future__ import annotations

import pytest

from config.constants import UNKNOWN_EVIDENCE_LEVEL
from src.generation.card_generator import _UNGRADED_LEVEL, _apply_intent_map, _labels
from src.ingestion.chunk import EVIDENCE_LEVELS


# ── Missing evidence grade ───────────────────────────────────────────────────

class TestMissingGradeIsNotB:
    def test_ungraded_default_is_the_weakest_enum_value(self):
        """`level` used to default to "B" -- a guideline-endorsed grade backed
        by cohort evidence. A missing grade is not a B grade."""
        assert _UNGRADED_LEVEL == "Expert opinion"
        assert _UNGRADED_LEVEL != "B"

    def test_ungraded_default_is_inside_the_tool_enum(self):
        """Anything outside the enum would render as an unstyled tag and would
        not round-trip through the tool schema."""
        from src.generation.card_generator import _build_card_tool

        schema = _build_card_tool(True)["input_schema"]
        enum = schema["properties"]["treatment"]["items"]["properties"]["level"]["enum"]
        assert _UNGRADED_LEVEL in enum


# ── Missing therapeutic intent ───────────────────────────────────────────────

class TestMissingIntentIsNotPalliative:
    @pytest.mark.parametrize("language,forbidden", [("fr", "Palliatif"), ("en", "Palliative")])
    def test_default_no_longer_asserts_incurable_disease(self, language, forbidden):
        """Defaulting to palliative asserts the disease is incurable. Of
        everything on a treatment card, that is the most consequential thing to
        guess."""
        assert _labels(language)["default_intent"] != forbidden

    @pytest.mark.parametrize("language,expected", [("fr", "Non précisé"), ("en", "Not specified")])
    def test_default_says_it_is_unspecified(self, language, expected):
        assert _labels(language)["default_intent"] == expected


# ── Intent map application ───────────────────────────────────────────────────

class TestIntentMatchingRefusesAmbiguity:
    @pytest.mark.parametrize(
        "drug,key",
        [
            # Each of these matched under the old `drug in key` arm. Verified
            # against the previous implementation: a bare drug name that is a
            # substring of some other treatment's key took that key's intent.
            ("Prednisone", "Abiraterone + prednisone"),
            ("BCG", "BCG maintenance therapy"),
            ("Docetaxel", "Docetaxel + ADT"),
        ],
    )
    def test_bare_drug_name_no_longer_matches_a_longer_unrelated_key(self, drug, key):
        treatments = [{"drug": drug, "intent": "Not specified"}]
        _apply_intent_map(treatments, {key: "Palliative"})
        assert treatments[0]["intent"] == "Not specified", (
            f"{drug!r} should not inherit the intent assigned to {key!r}"
        )

    def test_the_old_bidirectional_arm_would_have_matched_these(self):
        """Guards the cases above: if the drug/key pairs stopped exercising the
        old failure, the tests would pass vacuously."""
        for drug, key in [
            ("Prednisone", "Abiraterone + prednisone"),
            ("BCG", "BCG maintenance therapy"),
            ("Docetaxel", "Docetaxel + ADT"),
        ]:
            old_match = key.lower() in drug.lower() or drug.lower() in key.lower()
            assert old_match, f"{drug!r}/{key!r} no longer reproduces the old bug"

    def test_exact_match_still_applies(self):
        treatments = [{"drug": "Abiraterone", "intent": "Not specified"}]
        _apply_intent_map(treatments, {"Abiraterone": "Palliative"})
        assert treatments[0]["intent"] == "Palliative"

    def test_exact_match_ignores_case_and_padding(self):
        treatments = [{"drug": "  ABIRATERONE  ", "intent": "x"}]
        _apply_intent_map(treatments, {"abiraterone": "Curative"})
        assert treatments[0]["intent"] == "Curative"

    def test_key_contained_in_a_dosage_bearing_drug_string_applies(self):
        """The legitimate containment case: the card carries a dosage and a
        citation that the intent call never saw."""
        treatments = [{"drug": "Abiraterone 1000 mg/day [Doc 2]", "intent": "x"}]
        _apply_intent_map(treatments, {"Abiraterone": "Palliative"})
        assert treatments[0]["intent"] == "Palliative"

    def test_two_keys_matching_one_drug_assigns_neither(self):
        """Previously the first key in dict order won."""
        treatments = [{"drug": "abiraterone plus prednisone", "intent": "Not specified"}]
        _apply_intent_map(
            treatments,
            {"abiraterone": "Palliative", "prednisone": "Adjuvant"},
        )
        assert treatments[0]["intent"] == "Not specified"

    def test_two_keys_agreeing_still_applies(self):
        treatments = [{"drug": "abiraterone plus prednisone", "intent": "x"}]
        _apply_intent_map(
            treatments,
            {"abiraterone": "Palliative", "prednisone": "Palliative"},
        )
        assert treatments[0]["intent"] == "Palliative"

    def test_each_treatment_keeps_its_own_intent(self):
        treatments = [
            {"drug": "Radical prostatectomy", "intent": "x"},
            {"drug": "Abiraterone 1000 mg/day", "intent": "x"},
        ]
        _apply_intent_map(
            treatments,
            {"Radical prostatectomy": "Curative", "Abiraterone": "Palliative"},
        )
        assert [t["intent"] for t in treatments] == ["Curative", "Palliative"]

    def test_empty_map_is_a_noop(self):
        treatments = [{"drug": "Abiraterone", "intent": "Not specified"}]
        _apply_intent_map(treatments, {})
        assert treatments[0]["intent"] == "Not specified"


# ── Missing evidence level ───────────────────────────────────────────────────

class TestUnknownEvidenceLevelIsShared:
    def test_matches_the_ingestion_mapping(self):
        assert UNKNOWN_EVIDENCE_LEVEL == EVIDENCE_LEVELS["unknown"]

    def test_confidence_and_judges_agree(self):
        """confidence.py defaulted to 6 and judges.py to 1, for the same
        missing field, so one treated an unlabelled chunk as the weakest
        evidence and the other as the strongest."""
        import inspect

        from src.evaluation import judges
        from src.generation import confidence

        for module in (confidence, judges):
            src = inspect.getsource(module)
            assert 'get("evidence_level", UNKNOWN_EVIDENCE_LEVEL)' in src, module.__name__

    def test_unlabelled_chunk_now_requires_hedging(self):
        """Chunks with no evidence_level are mostly PubMed web-fallback
        results: fetched live, never reranked, never quality-gated. Treating
        them as high-evidence waived the hedging requirement entirely."""
        from src.evaluation.judges import _heuristic_evidence_appropriate

        class _Chunk:
            text = "some finding"
            metadata: dict = {}

        unhedged = "Abiraterone improves overall survival."
        score, _ = _heuristic_evidence_appropriate(unhedged, [_Chunk()])
        assert score == 0.7

        hedged = "Limited evidence suggests abiraterone may improve survival."
        assert _heuristic_evidence_appropriate(hedged, [_Chunk()])[0] == 1.0
