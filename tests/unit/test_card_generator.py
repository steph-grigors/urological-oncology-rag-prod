"""
Unit tests for src/generation/card_generator.py.

All LLM calls are mocked. Focuses on:
  - _strip_doc_tags
  - _format_comorbidities / _build_context_block
  - _apply_regulatory_to_triplets (French warnings, indication-gated)
  - _apply_biomarker_to_triplets (fires when biomarker absent)
  - CardGenerator.generate_card (full pipeline with mocked tool_use)
  - CardGenerator._reclassify_intent (intent map application)
  - translate_to_english (passthrough on failure)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.generation.card_generator import (
    CardGenerator,
    TreatmentTriplet,
    TreatmentWarning,
    _apply_biomarker_to_triplets,
    _apply_regulatory_to_triplets,
    _build_context_block,
    _format_comorbidities,
    _strip_doc_tags,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_chunk(title: str = "Test Study", text: str = "Study text.", year: int = 2023) -> MagicMock:
    chunk = MagicMock()
    chunk.text = text
    chunk.metadata = {"title": title, "year": year, "study_design": "rct", "sample_size": 100}
    return chunk


_UNSET = object()  # sentinel so tool_return=None means "return None", not "use default"

_DEFAULT_TOOL_RETURN = {
    "input": {
        "stage": "cT3b N1 M1b, ISUP 4, PSA 42 ng/mL",
        "confidence": "High",
        "guideline": "EAU 2024",
        "comorbidities_impact": "Moderate renal impairment — adjust dosage.",
        "treatment": [
            {"drug": "ADT + Abiraterone 1000 mg/day", "intent": "Palliative", "level": "A"},
            {"drug": "ADT + Docetaxel 75 mg/m²", "intent": "Palliative", "level": "A"},
        ],
        "treatment_confidence": "High",
        "sources": ["Fizazi et al. 2017 NEJM (RCT, n=1199)"],
    },
    "prompt_tokens": 100,
    "completion_tokens": 200,
}


def _make_llm(
    complete_return: str = "English query",
    tool_return: object = _UNSET,
) -> MagicMock:
    llm = MagicMock()
    llm.provider = "anthropic"
    llm.model = "claude-sonnet-4-6"
    llm.complete.return_value = MagicMock(content=complete_return)
    llm.complete_with_tools.return_value = (
        _DEFAULT_TOOL_RETURN if tool_return is _UNSET else tool_return
    )
    return llm


_ATEZOLIZUMAB_ENTRY = {
    "drug": "atezolizumab",
    "aliases": ["tecentriq"],
    "indication_keywords": ["urothelial", "bladder", "platinum-ineligible"],
    "jurisdiction": "EMA",
    "status": "withdrawn",
    "warning": "Atezolizumab EMA approval withdrawn (2021).",
    "warning_fr": "Atézolizumab — AMM EMA retirée (2021).",
    "source": "manual",
}

_LUTETIUM_BIOMARKER = {
    "drug_patterns": ["lu-177", "lutetium", "pluvicto"],
    "required_biomarker": "PSMA-PET positivity",
    "biomarker_keywords": ["psma-pet", "psma positif", "psma positive"],
    "warning_fr": "Lu-177 requiert une positivité PSMA-PET confirmée.",
    "jurisdiction": "biomarker",
}

_PARP_BIOMARKER = {
    "drug_patterns": ["olaparib", "lynparza", "rucaparib"],
    "required_biomarker": "HRR/BRCA",
    "biomarker_keywords": ["brca", "hrr"],
    "warning_fr": "Inhibiteur PARP requiert altération HRR/BRCA confirmée.",
    "jurisdiction": "biomarker",
}


# ── _strip_doc_tags ───────────────────────────────────────────────────────────

class TestStripDocTags:
    def test_inline_tag_removed(self):
        assert _strip_doc_tags("ADT + Abiratérone [Doc 1]") == "ADT + Abiratérone"

    def test_tag_with_leading_space(self):
        assert _strip_doc_tags("Some text [Doc 3]") == "Some text"

    def test_multiple_tags_removed(self):
        assert _strip_doc_tags("Drug A [Doc 1] and Drug B [Doc 2]") == "Drug A and Drug B"

    def test_no_tag_unchanged(self):
        assert _strip_doc_tags("Clean text") == "Clean text"

    def test_empty_string(self):
        assert _strip_doc_tags("") == ""


# ── _format_comorbidities ─────────────────────────────────────────────────────

class TestFormatComorbidities:
    def test_empty_returns_default(self):
        assert _format_comorbidities({}) == "Aucune comorbidité précisée"

    def test_single_entry(self):
        result = _format_comorbidities({"IRC": "oui"})
        assert "IRC: oui" in result

    def test_multiple_entries(self):
        result = _format_comorbidities({"IRC": "oui", "Diabète": "non"})
        assert "IRC: oui" in result
        assert "Diabète: non" in result


# ── _build_context_block ──────────────────────────────────────────────────────

class TestBuildContextBlock:
    def test_numbered_doc_headers(self):
        chunks = [_make_chunk("Study A"), _make_chunk("Study B")]
        result = _build_context_block(chunks)
        assert "[Doc 1]" in result
        assert "[Doc 2]" in result
        assert "Study A" in result
        assert "Study B" in result

    def test_text_included(self):
        chunk = _make_chunk(text="Survival benefit observed.")
        result = _build_context_block([chunk])
        assert "Survival benefit observed." in result

    def test_empty_chunks(self):
        assert _build_context_block([]) == ""

    def test_truncation_at_max_chars(self):
        chunk = _make_chunk(text="x" * 10000)
        result = _build_context_block([chunk], max_chars=500)
        assert len(result) <= 510  # small tolerance for header


# ── _apply_regulatory_to_triplets ─────────────────────────────────────────────

class TestApplyRegulatoryToTriplets:
    def test_warning_fires_on_drug_and_indication(self):
        triplets = [TreatmentTriplet(drug="atezolizumab", intent="Palliatif", level="A")]
        context = "urothelial carcinoma platinum-ineligible patient"
        with patch("src.generation.card_generator._load_withdrawals", return_value=(_ATEZOLIZUMAB_ENTRY,)):
            _apply_regulatory_to_triplets(triplets, context)
        assert len(triplets[0].warnings) == 1
        assert triplets[0].warnings[0].type == "regulatory"
        assert "AMM EMA retirée" in triplets[0].warnings[0].message

    def test_alias_triggers_warning(self):
        triplets = [TreatmentTriplet(drug="tecentriq", intent="Palliatif", level="A")]
        context = "bladder cancer platinum-ineligible"
        with patch("src.generation.card_generator._load_withdrawals", return_value=(_ATEZOLIZUMAB_ENTRY,)):
            _apply_regulatory_to_triplets(triplets, context)
        assert len(triplets[0].warnings) == 1

    def test_indication_absent_no_warning(self):
        triplets = [TreatmentTriplet(drug="atezolizumab", intent="Palliatif", level="A")]
        context = "non-small cell lung cancer patient"
        with patch("src.generation.card_generator._load_withdrawals", return_value=(_ATEZOLIZUMAB_ENTRY,)):
            _apply_regulatory_to_triplets(triplets, context)
        assert triplets[0].warnings == []

    def test_uses_warning_fr_when_available(self):
        triplets = [TreatmentTriplet(drug="atezolizumab", intent="Palliatif", level="A")]
        context = "urothelial platinum-ineligible"
        with patch("src.generation.card_generator._load_withdrawals", return_value=(_ATEZOLIZUMAB_ENTRY,)):
            _apply_regulatory_to_triplets(triplets, context)
        assert "AMM EMA retirée" in triplets[0].warnings[0].message  # French text

    def test_unrelated_drug_no_warning(self):
        triplets = [TreatmentTriplet(drug="enzalutamide", intent="Palliatif", level="A")]
        context = "castration-resistant prostate cancer"
        with patch("src.generation.card_generator._load_withdrawals", return_value=(_ATEZOLIZUMAB_ENTRY,)):
            _apply_regulatory_to_triplets(triplets, context)
        assert triplets[0].warnings == []

    def test_no_entries_no_warnings(self):
        triplets = [TreatmentTriplet(drug="atezolizumab", intent="Palliatif", level="A")]
        with patch("src.generation.card_generator._load_withdrawals", return_value=()):
            _apply_regulatory_to_triplets(triplets, "urothelial")
        assert triplets[0].warnings == []


# ── _apply_biomarker_to_triplets ──────────────────────────────────────────────

class TestApplyBiomarkerToTriplets:
    def test_fires_when_biomarker_absent(self):
        triplets = [TreatmentTriplet(drug="lu-177-psma-617", intent="Palliatif", level="B")]
        context = "castration-resistant prostate cancer, PSA 42"
        with patch("src.generation.card_generator._load_biomarker_entries", return_value=(_LUTETIUM_BIOMARKER,)):
            _apply_biomarker_to_triplets(triplets, context)
        assert len(triplets[0].warnings) == 1
        assert triplets[0].warnings[0].type == "biomarker"
        assert "PSMA-PET" in triplets[0].warnings[0].message

    def test_no_warning_when_biomarker_present(self):
        triplets = [TreatmentTriplet(drug="lu-177", intent="Palliatif", level="B")]
        context = "psma-pet positif, psma positive SUVmax > seuil hépatique"
        with patch("src.generation.card_generator._load_biomarker_entries", return_value=(_LUTETIUM_BIOMARKER,)):
            _apply_biomarker_to_triplets(triplets, context)
        assert triplets[0].warnings == []

    def test_parp_fires_without_brca(self):
        triplets = [TreatmentTriplet(drug="olaparib", intent="Palliatif", level="A")]
        context = "castration-resistant prostate cancer, mCRPC"
        with patch("src.generation.card_generator._load_biomarker_entries", return_value=(_PARP_BIOMARKER,)):
            _apply_biomarker_to_triplets(triplets, context)
        assert len(triplets[0].warnings) == 1
        assert "HRR/BRCA" in triplets[0].warnings[0].message

    def test_parp_silent_when_brca_mentioned(self):
        triplets = [TreatmentTriplet(drug="olaparib", intent="Palliatif", level="A")]
        context = "brca2 mutation detected, mCRPC"
        with patch("src.generation.card_generator._load_biomarker_entries", return_value=(_PARP_BIOMARKER,)):
            _apply_biomarker_to_triplets(triplets, context)
        assert triplets[0].warnings == []

    def test_unrelated_drug_no_biomarker_warning(self):
        triplets = [TreatmentTriplet(drug="enzalutamide 160 mg/j", intent="Palliatif", level="A")]
        context = "mCRPC, PSA 88 ng/mL"
        with patch("src.generation.card_generator._load_biomarker_entries", return_value=(_LUTETIUM_BIOMARKER,)):
            _apply_biomarker_to_triplets(triplets, context)
        assert triplets[0].warnings == []


# ── CardGenerator.translate_to_english ───────────────────────────────────────

class TestTranslateToEnglish:
    def test_returns_translated_text(self):
        llm = _make_llm(complete_return="metastatic prostate cancer first line")
        gen = CardGenerator(llm_client=llm)
        result = gen.translate_to_english("Cancer de prostate métastatique première ligne")
        assert result == "metastatic prostate cancer first line"

    def test_falls_back_on_error(self):
        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("API down")
        gen = CardGenerator(llm_client=llm)
        original = "Cancer de prostate"
        result = gen.translate_to_english(original)
        assert result == original


# ── CardGenerator.generate_card (full pipeline) ───────────────────────────────

class TestGenerateCard:
    def _run(self, llm: MagicMock, patient_context: str = "prostate mCRPC") -> object:
        gen = CardGenerator(llm_client=llm)
        chunks = [_make_chunk()]
        with (
            patch("src.generation.card_generator._load_withdrawals", return_value=()),
            patch("src.generation.card_generator._load_biomarker_entries", return_value=()),
        ):
            return gen.generate_card(
                patient_id="P1",
                cancer_type="prostate",
                age_range="70–79 ans",
                clinical_history=patient_context,
                comorbidities={"IRC": "oui"},
                ranked_chunks=chunks,
                confidence_score=0.82,
            )

    def test_patient_id_preserved(self):
        result = self._run(_make_llm())
        assert result.patient_id == "P1"

    def test_doc_tags_stripped_from_stage(self):
        llm = _make_llm(
            tool_return={
                "input": {
                    "stage": "cT3b N1 [Doc 1]",
                    "confidence": "Élevée",
                    "guideline": "EAU 2024",
                    "comorbidities_impact": "Modérée.",
                    "treatment": [{"drug": "ADT [Doc 1]", "intent": "Palliatif", "level": "A"}],
                    "treatment_confidence": "Élevée",
                    "sources": ["Ref [Doc 1]"],
                },
                "prompt_tokens": 10,
                "completion_tokens": 20,
            }
        )
        result = self._run(llm)
        assert "[Doc" not in result.stage
        assert "[Doc" not in result.treatment[0].drug
        assert all("[Doc" not in s for s in result.sources)

    def test_treatment_triplets_built(self):
        result = self._run(_make_llm())
        assert len(result.treatment) == 2
        assert result.treatment[0].drug == "ADT + Abiraterone 1000 mg/day"
        assert result.treatment[0].level == "A"

    def test_retrieval_metadata_populated(self):
        result = self._run(_make_llm())
        assert result.retrieval_metadata["chunks_used"] == 1
        assert result.retrieval_metadata["confidence_score"] == 0.82

    def test_tool_call_failure_returns_empty_card(self):
        llm = _make_llm(tool_return=None)
        # complete_with_tools returns None → card_data is {}
        result = self._run(llm)
        assert result.patient_id == "P1"
        assert result.treatment == []

    def test_s1_intent_reclassification_applied(self):
        llm = _make_llm()
        llm.complete.return_value = MagicMock(
            content='{"ADT + Abiraterone 1000 mg/day": "Palliative", "ADT + Docetaxel 75 mg/m²": "Palliative"}'
        )
        result = self._run(llm)
        # S1 call should have been made and intents applied
        assert llm.complete.called
        assert all(t.intent in ("Curative", "Palliative", "Adjuvant") for t in result.treatment)


# ── Warning integration: regulatory fires correctly in full pipeline ───────────

class TestWarningIntegration:
    def test_regulatory_warning_added_for_atezolizumab(self):
        llm = _make_llm(
            tool_return={
                "input": {
                    "stage": "cT3 N0 M0",
                    "confidence": "Moderate",
                    "guideline": "EAU 2024",
                    "comorbidities_impact": "None.",
                    "treatment": [{"drug": "atezolizumab", "intent": "Palliative", "level": "B"}],
                    "treatment_confidence": "Moderate",
                    "sources": [],
                },
                "prompt_tokens": 10,
                "completion_tokens": 20,
            }
        )
        gen = CardGenerator(llm_client=llm)
        with (
            patch("src.generation.card_generator._load_withdrawals", return_value=(_ATEZOLIZUMAB_ENTRY,)),
            patch("src.generation.card_generator._load_biomarker_entries", return_value=()),
        ):
            result = gen.generate_card(
                patient_id="P2",
                cancer_type="bladder",
                age_range="",
                clinical_history="platinum-ineligible urothelial carcinoma",
                comorbidities={},
                ranked_chunks=[_make_chunk()],
                confidence_score=0.6,
            )
        assert any(w.type == "regulatory" for w in result.treatment[0].warnings)
        assert "AMM EMA retirée" in result.treatment[0].warnings[0].message

    def test_biomarker_warning_added_for_lutetium(self):
        llm = _make_llm(
            tool_return={
                "input": {
                    "stage": "mCRPC",
                    "confidence": "High",
                    "guideline": "EAU 2024",
                    "comorbidities_impact": "None.",
                    "treatment": [{"drug": "lu-177-psma-617", "intent": "Palliative", "level": "A"}],
                    "treatment_confidence": "High",
                    "sources": [],
                },
                "prompt_tokens": 10,
                "completion_tokens": 20,
            }
        )
        gen = CardGenerator(llm_client=llm)
        with (
            patch("src.generation.card_generator._load_withdrawals", return_value=()),
            patch("src.generation.card_generator._load_biomarker_entries", return_value=(_LUTETIUM_BIOMARKER,)),
        ):
            result = gen.generate_card(
                patient_id="P3",
                cancer_type="prostate",
                age_range="",
                clinical_history="mCRPC post-docétaxel, PSA 88 ng/mL",
                comorbidities={},
                ranked_chunks=[_make_chunk()],
                confidence_score=0.8,
            )
        assert any(w.type == "biomarker" for w in result.treatment[0].warnings)
