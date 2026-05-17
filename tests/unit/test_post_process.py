"""
Unit tests for src/generation/post_process.py.

All tests mock _load_withdrawals so no real JSON file is required.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.generation.post_process import apply_regulatory_warnings

# ── Fixtures ──────────────────────────────────────────────────────────────────

_ATEZOLIZUMAB_ENTRY = {
    "drug": "atezolizumab",
    "aliases": ["tecentriq"],
    "indication_keywords": ["urothelial", "bladder", "platinum-ineligible"],
    "jurisdiction": "EMA",
    "status": "withdrawn",
    "warning": "Atezolizumab EMA approval for platinum-ineligible UC was withdrawn (2021).",
    "source": "manual",
}

_RUCAPARIB_ENTRY = {
    "drug": "rucaparib",
    "aliases": ["rubraca"],
    "indication_keywords": ["prostate", "crpc", "castration-resistant"],
    "jurisdiction": "FDA",
    "status": "withdrawn",
    "warning": "Rucaparib FDA approval for BRCA-mutated CRPC was withdrawn (2022).",
    "source": "manual",
}

_NO_INDICATION_ENTRY = {
    "drug": "testdrug",
    "aliases": [],
    "indication_keywords": [],
    "jurisdiction": "EMA",
    "status": "withdrawn",
    "warning": "Testdrug always triggers when mentioned.",
    "source": "manual",
}

_TEST_ENTRIES = (_ATEZOLIZUMAB_ENTRY, _RUCAPARIB_ENTRY)


# ── Warning fires correctly ────────────────────────────────────────────────────

class TestWarningFires:
    def test_drug_and_indication_both_present(self):
        answer = (
            "Atezolizumab demonstrated OS benefit in platinum-ineligible "
            "urothelial carcinoma patients [Doc 1]."
        )
        with patch("src.generation.post_process._load_withdrawals", return_value=_TEST_ENTRIES):
            result = apply_regulatory_warnings(answer)
        assert "⚠️" in result
        assert "Regulatory note" in result
        assert "Atezolizumab EMA approval" in result

    def test_alias_triggers_warning(self):
        answer = "Tecentriq showed activity in bladder cancer [Doc 1]."
        with patch("src.generation.post_process._load_withdrawals", return_value=_TEST_ENTRIES):
            result = apply_regulatory_warnings(answer)
        assert "Atezolizumab EMA approval" in result

    def test_case_insensitive_drug_match(self):
        answer = "ATEZOLIZUMAB was evaluated in urothelial carcinoma [Doc 1]."
        with patch("src.generation.post_process._load_withdrawals", return_value=_TEST_ENTRIES):
            result = apply_regulatory_warnings(answer)
        assert "⚠️" in result

    def test_case_insensitive_indication_match(self):
        answer = "Atezolizumab in UROTHELIAL carcinoma patients [Doc 1]."
        with patch("src.generation.post_process._load_withdrawals", return_value=_TEST_ENTRIES):
            result = apply_regulatory_warnings(answer)
        assert "⚠️" in result

    def test_second_drug_fires_independently(self):
        answer = "Rucaparib improved rPFS in castration-resistant prostate cancer [Doc 1]."
        with patch("src.generation.post_process._load_withdrawals", return_value=_TEST_ENTRIES):
            result = apply_regulatory_warnings(answer)
        assert "Rucaparib FDA approval" in result

    def test_entry_without_indication_keywords_always_fires(self):
        entries = (_NO_INDICATION_ENTRY,)
        answer = "Testdrug improved outcomes [Doc 1]."
        with patch("src.generation.post_process._load_withdrawals", return_value=entries):
            result = apply_regulatory_warnings(answer)
        assert "Testdrug always triggers" in result

    def test_multiple_matches_produce_multiple_warnings(self):
        answer = (
            "Atezolizumab and rucaparib were evaluated in bladder and "
            "castration-resistant prostate cancer patients."
        )
        with patch("src.generation.post_process._load_withdrawals", return_value=_TEST_ENTRIES):
            result = apply_regulatory_warnings(answer)
        assert "Atezolizumab EMA approval" in result
        assert "Rucaparib FDA approval" in result


# ── Warning does not fire ──────────────────────────────────────────────────────

class TestWarningDoesNotFire:
    def test_drug_present_but_indication_absent(self):
        # Atezolizumab mentioned for NSCLC — no urothelial/bladder context
        answer = "Atezolizumab demonstrated activity in non-small cell lung cancer [Doc 1]."
        with patch("src.generation.post_process._load_withdrawals", return_value=_TEST_ENTRIES):
            result = apply_regulatory_warnings(answer)
        assert "⚠️" not in result
        assert result == answer  # unchanged

    def test_no_matching_drug(self):
        answer = "Enzalutamide significantly improved overall survival in mCRPC [Doc 1]."
        with patch("src.generation.post_process._load_withdrawals", return_value=_TEST_ENTRIES):
            result = apply_regulatory_warnings(answer)
        assert "⚠️" not in result
        assert result == answer

    def test_indication_present_but_drug_absent(self):
        answer = "Pembrolizumab was evaluated in urothelial carcinoma [Doc 1]."
        with patch("src.generation.post_process._load_withdrawals", return_value=_TEST_ENTRIES):
            result = apply_regulatory_warnings(answer)
        assert "⚠️" not in result

    def test_empty_answer_unchanged(self):
        with patch("src.generation.post_process._load_withdrawals", return_value=_TEST_ENTRIES):
            result = apply_regulatory_warnings("")
        assert result == ""

    def test_no_entries_loaded(self):
        answer = "Atezolizumab in urothelial carcinoma [Doc 1]."
        with patch("src.generation.post_process._load_withdrawals", return_value=()):
            result = apply_regulatory_warnings(answer)
        assert result == answer


# ── Answer content preserved ──────────────────────────────────────────────────

class TestAnswerPreserved:
    def test_original_text_present_in_output(self):
        original = "Atezolizumab improved PFS in urothelial carcinoma patients [Doc 1]."
        with patch("src.generation.post_process._load_withdrawals", return_value=_TEST_ENTRIES):
            result = apply_regulatory_warnings(original)
        assert original in result

    def test_warning_appended_not_prepended(self):
        answer = "Atezolizumab in urothelial carcinoma [Doc 1]."
        with patch("src.generation.post_process._load_withdrawals", return_value=_TEST_ENTRIES):
            result = apply_regulatory_warnings(answer)
        assert result.startswith(answer)

    def test_no_match_returns_identical_object(self):
        answer = "No relevant drugs mentioned."
        with patch("src.generation.post_process._load_withdrawals", return_value=_TEST_ENTRIES):
            result = apply_regulatory_warnings(answer)
        assert result is answer
