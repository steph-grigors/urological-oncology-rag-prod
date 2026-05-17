"""
Unit tests for scripts/update_regulatory_db.py.

All external calls (requests, anthropic) are mocked.
Also covers real-file loading of data/regulatory_withdrawals.json as a
regression guard — if the seeded JSON becomes malformed, this test fails.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the script as a module (it guards execution behind __main__)
from scripts.update_regulatory_db import (
    _is_oncology,
    collect_openfda_entries,
    load_existing,
    merge_entries,
    openfda_record_to_entry,
    query_openfda_enforcement,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _openfda_record(
    product_desc: str = "Tecentriq (atezolizumab) injection for urothelial carcinoma",
    reason: str = "Voluntary withdrawal of bladder cancer indication following post-marketing results",
    voluntary: str = "Voluntary",
    status: str = "Terminated",
    date: str = "20210915",
) -> dict:
    return {
        "product_description": product_desc,
        "reason_for_recall": reason,
        "voluntary_mandated": voluntary,
        "status": status,
        "recall_initiation_date": date,
    }


# ── _is_oncology ──────────────────────────────────────────────────────────────

class TestIsOncology:
    def test_bladder_keyword(self):
        assert _is_oncology({"product_description": "bladder cancer drug"})

    def test_carcinoma_keyword(self):
        assert _is_oncology({"reason_for_recall": "urothelial carcinoma indication"})

    def test_non_oncology_returns_false(self):
        assert not _is_oncology({
            "product_description": "ibuprofen tablets 200mg",
            "reason_for_recall": "incorrect labeling"
        })

    def test_combined_fields(self):
        assert _is_oncology({
            "product_description": "Tecentriq injection",
            "reason_for_recall": "cancer treatment withdrawal",
        })


# ── openfda_record_to_entry ───────────────────────────────────────────────────

class TestOpenFDARecordToEntry:
    def test_valid_voluntary_oncology_record(self):
        rec = _openfda_record()
        entry = openfda_record_to_entry("atezolizumab", rec)
        assert entry is not None
        assert entry["drug"] == "atezolizumab"
        assert entry["jurisdiction"] == "FDA"
        assert entry["status"] == "withdrawn"  # Terminated → withdrawn
        assert entry["date"] == "2021-09"
        assert "Verify current regulatory status" in entry["warning"]

    def test_non_voluntary_returns_none(self):
        rec = _openfda_record(voluntary="Mandatory")
        assert openfda_record_to_entry("atezolizumab", rec) is None

    def test_non_oncology_returns_none(self):
        rec = _openfda_record(
            product_desc="Paracetamol tablets 500mg",
            reason="Incorrect dosage on label",
        )
        assert openfda_record_to_entry("paracetamol", rec) is None

    def test_ongoing_recall_not_withdrawn(self):
        rec = _openfda_record(status="Ongoing")
        entry = openfda_record_to_entry("atezolizumab", rec)
        assert entry is not None
        assert entry["status"] == "recalled"  # Ongoing → recalled (not withdrawn)

    def test_date_parsing(self):
        rec = _openfda_record(date="20221001")
        entry = openfda_record_to_entry("rucaparib", rec)
        assert entry["date"] == "2022-10"

    def test_malformed_date_handled(self):
        rec = _openfda_record(date="invalid")
        entry = openfda_record_to_entry("atezolizumab", rec)
        assert entry is not None  # Should not raise


# ── query_openfda_enforcement ─────────────────────────────────────────────────

class TestQueryOpenFDA:
    def test_returns_results_on_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": [_openfda_record()]}
        with patch("scripts.update_regulatory_db.requests.get", return_value=mock_resp):
            results = query_openfda_enforcement("atezolizumab")
        assert len(results) == 1

    def test_returns_empty_on_404(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status.side_effect = Exception("404 Not Found")

        import requests as req
        http_err = req.HTTPError(response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_err

        with patch("scripts.update_regulatory_db.requests.get", return_value=mock_resp):
            results = query_openfda_enforcement("unknowndrug")
        assert results == []

    def test_returns_empty_on_connection_error(self):
        import requests as req
        with patch(
            "scripts.update_regulatory_db.requests.get",
            side_effect=req.ConnectionError("timeout"),
        ):
            results = query_openfda_enforcement("atezolizumab")
        assert results == []


# ── merge_entries ─────────────────────────────────────────────────────────────

class TestMergeEntries:
    _MANUAL_ENTRY = {
        "drug": "atezolizumab",
        "jurisdiction": "EMA",
        "status": "withdrawn",
        "warning": "Manual warning.",
        "source": "manual",
    }
    _AUTO_ENTRY = {
        "drug": "rucaparib",
        "jurisdiction": "FDA",
        "status": "withdrawn",
        "warning": "Auto warning.",
        "source": "openfda_api",
    }

    def test_manual_entry_not_overwritten(self):
        new = [dict(self._MANUAL_ENTRY, warning="New automated warning.", source="openfda_api")]
        result = merge_entries([self._MANUAL_ENTRY], new)
        entry = next(e for e in result if e["drug"] == "atezolizumab")
        assert entry["warning"] == "Manual warning."

    def test_automated_entry_overwritten_by_fresher_data(self):
        new = [dict(self._AUTO_ENTRY, warning="Updated auto warning.")]
        result = merge_entries([self._AUTO_ENTRY], new)
        entry = next(e for e in result if e["drug"] == "rucaparib")
        assert entry["warning"] == "Updated auto warning."

    def test_new_entry_added(self):
        new_entry = {
            "drug": "olaparib",
            "jurisdiction": "FDA",
            "status": "withdrawn",
            "warning": "Olaparib withdrawn.",
            "source": "openfda_api",
        }
        result = merge_entries([self._MANUAL_ENTRY], [new_entry])
        drugs = {e["drug"] for e in result}
        assert "atezolizumab" in drugs
        assert "olaparib" in drugs

    def test_entry_without_warning_skipped(self):
        bad = {"drug": "testdrug", "jurisdiction": "FDA", "warning": "", "source": "openfda_api"}
        result = merge_entries([], [bad])
        assert result == []

    def test_entry_without_drug_skipped(self):
        bad = {"drug": "", "jurisdiction": "FDA", "warning": "Some warning.", "source": "openfda_api"}
        result = merge_entries([], [bad])
        assert result == []

    def test_deduplication_by_drug_and_jurisdiction(self):
        dup = [
            dict(self._AUTO_ENTRY, warning="First."),
            dict(self._AUTO_ENTRY, warning="Second."),
        ]
        result = merge_entries([], dup)
        matching = [e for e in result if e["drug"] == "rucaparib"]
        assert len(matching) == 1


# ── Real-file loading regression guard ───────────────────────────────────────

class TestRealFileLoading:
    """Regression guard — ensures data/regulatory_withdrawals.json remains valid
    and contains the expected seed entries.  Fails immediately if the file is
    malformed or a manual entry is accidentally removed."""

    @pytest.fixture(autouse=True)
    def clear_lru_cache(self):
        from src.generation.post_process import _load_withdrawals
        _load_withdrawals.cache_clear()
        yield
        _load_withdrawals.cache_clear()

    def test_json_loads_without_error(self):
        from src.generation.post_process import _load_withdrawals
        entries = _load_withdrawals()
        assert isinstance(entries, tuple)
        assert len(entries) > 0, "regulatory_withdrawals.json must have at least one entry"

    def test_all_entries_have_required_keys(self):
        from src.generation.post_process import _load_withdrawals
        entries = _load_withdrawals()
        for entry in entries:
            assert "drug" in entry, f"Missing 'drug' in entry: {entry}"
            assert "warning" in entry, f"Missing 'warning' in entry: {entry}"
            assert entry["warning"].strip(), f"Empty warning in entry: {entry}"

    def test_atezolizumab_ema_entry_present(self):
        from src.generation.post_process import _load_withdrawals
        entries = _load_withdrawals()
        ema_atez = [
            e for e in entries
            if "atezolizumab" in e.get("drug", "").lower()
            and e.get("jurisdiction", "").upper() == "EMA"
        ]
        assert ema_atez, "Seeded EMA atezolizumab entry must be present"

    def test_atezolizumab_fires_on_real_data(self):
        # Ensures the seeded entry actually triggers for a representative answer
        from src.generation.post_process import _load_withdrawals, apply_regulatory_warnings
        _load_withdrawals()  # populate cache from real file
        answer = (
            "Atezolizumab improved PFS in first-line platinum-ineligible "
            "urothelial carcinoma patients [Doc 1]."
        )
        result = apply_regulatory_warnings(answer)
        assert "⚠️" in result, (
            "Seeded atezolizumab EMA entry did not fire — check indication_keywords "
            "in regulatory_withdrawals.json"
        )
