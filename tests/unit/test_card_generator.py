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
    _format_grounded_source,
    _ground_sources,
    _strip_doc_tags,
    _strip_invalid_doc_tags,
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


# ── language="en" support ──────────────────────────────────────────────────────

class TestLanguageSupport:
    def test_default_language_unchanged(self):
        """No `language` arg passed → byte-identical to pre-existing French defaults."""
        llm = _make_llm(tool_return={"input": {}, "prompt_tokens": 0, "completion_tokens": 0})
        gen = CardGenerator(llm_client=llm)
        with (
            patch("src.generation.card_generator._load_withdrawals", return_value=()),
            patch("src.generation.card_generator._load_biomarker_entries", return_value=()),
        ):
            result = gen.generate_card(
                patient_id="P4",
                cancer_type="prostate",
                age_range="",
                clinical_history="mCRPC",
                comorbidities={},
                ranked_chunks=[],
                confidence_score=0.5,
            )
        assert result.confidence == "Modérée"
        assert result.treatment_confidence == "Modérée"

    def test_english_fallback_defaults(self):
        """language='en' switches the server-injected fallback text to English."""
        llm = _make_llm(tool_return={"input": {}, "prompt_tokens": 0, "completion_tokens": 0})
        gen = CardGenerator(llm_client=llm)
        with (
            patch("src.generation.card_generator._load_withdrawals", return_value=()),
            patch("src.generation.card_generator._load_biomarker_entries", return_value=()),
        ):
            result = gen.generate_card(
                patient_id="P5",
                cancer_type="prostate",
                age_range="",
                clinical_history="mCRPC",
                comorbidities={},
                ranked_chunks=[],
                confidence_score=0.5,
                language="en",
            )
        assert result.confidence == "Moderate"
        assert result.treatment_confidence == "Moderate"

    def test_english_comorbidities_label(self):
        assert _format_comorbidities({}, "No comorbidities specified") == "No comorbidities specified"

    def test_english_regulatory_warning_uses_warning_field(self):
        triplets = [TreatmentTriplet(drug="atezolizumab", intent="Palliative", level="A")]
        context = "urothelial platinum-ineligible"
        with patch("src.generation.card_generator._load_withdrawals", return_value=(_ATEZOLIZUMAB_ENTRY,)):
            _apply_regulatory_to_triplets(triplets, context, "en")
        assert "EMA approval withdrawn" in triplets[0].warnings[0].message

    def test_english_biomarker_falls_back_to_french_when_no_english_text(self):
        """If no English warning text exists yet, never silently drop the safety warning."""
        triplets = [TreatmentTriplet(drug="lu-177-psma-617", intent="Palliative", level="B")]
        context = "mCRPC, PSA 42"
        with patch("src.generation.card_generator._load_biomarker_entries", return_value=(_LUTETIUM_BIOMARKER,)):
            _apply_biomarker_to_triplets(triplets, context, "en")
        assert len(triplets[0].warnings) == 1
        assert "PSMA-PET" in triplets[0].warnings[0].message  # French text used as fallback


# ── sources_detail (additive, grounded sources) ────────────────────────────────

class TestSourcesDetail:
    def test_sources_detail_present_alongside_free_text_sources(self):
        llm = _make_llm()
        gen = CardGenerator(llm_client=llm)
        chunk = _make_chunk(title="Fizazi RCT", year=2019)
        chunk.chunk_id = "PMC123_results_0"
        with (
            patch("src.generation.card_generator._load_withdrawals", return_value=()),
            patch("src.generation.card_generator._load_biomarker_entries", return_value=()),
        ):
            result = gen.generate_card(
                patient_id="P6",
                cancer_type="prostate",
                age_range="",
                clinical_history="mCRPC",
                comorbidities={},
                ranked_chunks=[chunk],
                confidence_score=0.7,
            )
        # Old field untouched — still the LLM's free-text sources.
        assert result.sources == ["Fizazi et al. 2017 NEJM (RCT, n=1199)"]
        # New field is additive and grounded in the actual chunk metadata.
        detail = result.retrieval_metadata["sources_detail"]
        assert len(detail) == 1
        assert detail[0]["title"] == "Fizazi RCT"
        assert detail[0]["year"] == 2019
        assert detail[0]["chunk_id"] == "PMC123_results_0"


# ── _strip_invalid_doc_tags (drug field, range-validated) ──────────────────────

class TestStripInvalidDocTags:
    def test_valid_tag_kept(self):
        assert _strip_invalid_doc_tags("Abiraterone [Doc 1]", n_chunks=2) == "Abiraterone [Doc 1]"

    def test_out_of_range_tag_removed(self):
        assert _strip_invalid_doc_tags("Abiraterone [Doc 5]", n_chunks=2) == "Abiraterone"

    def test_zero_is_always_invalid(self):
        assert _strip_invalid_doc_tags("Drug [Doc 0]", n_chunks=3) == "Drug"

    def test_mixed_valid_and_invalid(self):
        result = _strip_invalid_doc_tags("Drug A [Doc 1] plus Drug B [Doc 9]", n_chunks=2)
        assert result == "Drug A [Doc 1] plus Drug B"

    def test_no_chunks_strips_everything(self):
        assert _strip_invalid_doc_tags("Drug [Doc 1]", n_chunks=0) == "Drug"


# ── _ground_sources / _format_grounded_source (hallucination-free citations) ──

class TestGroundSources:
    def _chunk(self, **meta_overrides):
        chunk = MagicMock()
        chunk.metadata = {
            "title": "LATITUDE",
            "authors": ["Fizazi K", "Tran N"],
            "journal": "NEJM",
            "year": 2017,
            "study_design": "rct",
            "sample_size": 1199,
            **meta_overrides,
        }
        return chunk

    def test_renders_text_purely_from_metadata(self):
        chunks = [self._chunk()]
        result = _ground_sources(["whatever the model wrote [Doc 1]"], chunks)
        assert result == ["[Doc 1] Fizazi et al. 2017 NEJM (rct, n=1199)"]

    def test_llm_authored_text_is_discarded(self):
        """Even a wrong year/journal in the LLM's text is irrelevant — only the
        [Doc N] pointer is read; the citation text itself is always regenerated."""
        chunks = [self._chunk()]
        result = _ground_sources(
            ["Totally wrong author 1999 Fake Journal [Doc 1]"], chunks
        )
        assert "Fizazi" in result[0]
        assert "2017" in result[0]
        assert "Wrong" not in result[0] and "Fake" not in result[0]

    def test_out_of_range_doc_dropped_entirely(self):
        chunks = [self._chunk()]
        result = _ground_sources(["[Doc 7]"], chunks)
        assert result == []

    def test_no_tag_at_all_dropped(self):
        chunks = [self._chunk()]
        result = _ground_sources(["Fizazi et al. 2017 NEJM"], chunks)
        assert result == []

    def test_duplicate_doc_numbers_deduplicated(self):
        chunks = [self._chunk()]
        result = _ground_sources(["[Doc 1]", "also [Doc 1] again"], chunks)
        assert len(result) == 1

    def test_multiple_valid_docs_preserve_order(self):
        chunks = [self._chunk(), self._chunk(title="Other", authors=["Smith J"], year=2020)]
        result = _ground_sources(["[Doc 2]", "[Doc 1]"], chunks)
        assert "[Doc 2]" in result[0]
        assert "[Doc 1]" in result[1]

    def test_empty_chunks_yields_empty(self):
        assert _ground_sources(["[Doc 1]"], []) == []


# ── keep_citations integration (generate_card) ──────────────────────────────────

class TestKeepCitations:
    def _run(self, tool_return: dict, ranked_chunks=None, **kwargs):
        llm = _make_llm(tool_return=tool_return)
        gen = CardGenerator(llm_client=llm)
        chunks = ranked_chunks if ranked_chunks is not None else [_make_chunk()]
        with (
            patch("src.generation.card_generator._load_withdrawals", return_value=()),
            patch("src.generation.card_generator._load_biomarker_entries", return_value=()),
        ):
            return gen.generate_card(
                patient_id="P1",
                cancer_type="prostate",
                age_range="",
                clinical_history="mCRPC",
                comorbidities={},
                ranked_chunks=chunks,
                confidence_score=0.8,
                **kwargs,
            )

    def test_default_still_strips_drug_citation(self):
        tool_return = {
            "input": {
                "stage": "mCRPC", "confidence": "High", "guideline": "EAU 2024",
                "comorbidities_impact": "None.",
                "treatment": [{"drug": "Abiraterone [Doc 1]", "intent": "Palliative", "level": "A"}],
                "treatment_confidence": "High", "sources": ["Fizazi et al. 2017 [Doc 1]"],
            },
            "prompt_tokens": 1, "completion_tokens": 1,
        }
        result = self._run(tool_return)  # keep_citations defaults to False
        assert "[Doc" not in result.treatment[0].drug
        assert "[Doc" not in result.sources[0]

    def test_keep_citations_preserves_valid_drug_tag(self):
        tool_return = {
            "input": {
                "stage": "mCRPC", "confidence": "High", "guideline": "EAU 2024",
                "comorbidities_impact": "None.",
                "treatment": [{"drug": "Abiraterone [Doc 1]", "intent": "Palliative", "level": "A"}],
                "treatment_confidence": "High", "sources": ["[Doc 1]"],
            },
            "prompt_tokens": 1, "completion_tokens": 1,
        }
        result = self._run(tool_return, keep_citations=True)
        assert result.treatment[0].drug == "Abiraterone [Doc 1]"

    def test_keep_citations_strips_out_of_range_drug_tag(self):
        tool_return = {
            "input": {
                "stage": "mCRPC", "confidence": "High", "guideline": "EAU 2024",
                "comorbidities_impact": "None.",
                "treatment": [{"drug": "Abiraterone [Doc 99]", "intent": "Palliative", "level": "A"}],
                "treatment_confidence": "High", "sources": [],
            },
            "prompt_tokens": 1, "completion_tokens": 1,
        }
        result = self._run(tool_return, keep_citations=True)
        assert "[Doc" not in result.treatment[0].drug

    def test_keep_citations_grounds_sources_field(self):
        chunk = _make_chunk(title="LATITUDE", year=2017)
        chunk.metadata["authors"] = ["Fizazi K"]
        tool_return = {
            "input": {
                "stage": "mCRPC", "confidence": "High", "guideline": "EAU 2024",
                "comorbidities_impact": "None.",
                "treatment": [{"drug": "Abiraterone [Doc 1]", "intent": "Palliative", "level": "A"}],
                "treatment_confidence": "High",
                "sources": ["Some made-up citation text [Doc 1]"],
            },
            "prompt_tokens": 1, "completion_tokens": 1,
        }
        result = self._run(tool_return, ranked_chunks=[chunk], keep_citations=True)
        assert result.sources == ["[Doc 1] Fizazi et al. 2017 (rct, n=100)"]

    def test_stage_guideline_comorbidities_always_stripped_even_with_keep_citations(self):
        tool_return = {
            "input": {
                "stage": "mCRPC [Doc 1]", "confidence": "High",
                "guideline": "EAU 2024 [Doc 1]",
                "comorbidities_impact": "None [Doc 1].",
                "treatment": [{"drug": "Abiraterone [Doc 1]", "intent": "Palliative", "level": "A"}],
                "treatment_confidence": "High", "sources": ["[Doc 1]"],
            },
            "prompt_tokens": 1, "completion_tokens": 1,
        }
        result = self._run(tool_return, keep_citations=True)
        assert "[Doc" not in result.stage
        assert "[Doc" not in result.guideline
        assert "[Doc" not in result.comorbidities_impact
        assert "[Doc" in result.treatment[0].drug  # only drug + sources keep tags


# ── disclose_fallback (parametric-knowledge disclosure) ─────────────────────────

class TestDiscloseFallback:
    def _run(self, ranked_chunks: list, **kwargs):
        llm = _make_llm()
        gen = CardGenerator(llm_client=llm)
        with (
            patch("src.generation.card_generator._load_withdrawals", return_value=()),
            patch("src.generation.card_generator._load_biomarker_entries", return_value=()),
        ):
            return gen.generate_card(
                patient_id="P1",
                cancer_type="prostate",
                age_range="",
                clinical_history="mCRPC",
                comorbidities={},
                ranked_chunks=ranked_chunks,
                confidence_score=0.1,
                **kwargs,
            )

    def test_default_does_not_touch_retrieval_metadata_shape(self):
        result = self._run([])  # disclose_fallback defaults to False
        assert "grounded" not in result.retrieval_metadata
        # sources stay whatever the LLM happened to write — no override
        assert result.sources == ["Fizazi et al. 2017 NEJM (RCT, n=1199)"]

    def test_disclose_fallback_replaces_sources_when_no_chunks(self):
        result = self._run([], disclose_fallback=True, language="en")
        assert result.retrieval_metadata["grounded"] is False
        assert len(result.sources) == 1
        assert "No relevant literature" in result.sources[0]
        assert result.retrieval_metadata["sources_detail"][0]["study_design"] == "parametric_knowledge"

    def test_disclose_fallback_french_text_by_default_language(self):
        result = self._run([], disclose_fallback=True, language="fr")
        assert "Aucune littérature pertinente" in result.sources[0]

    def test_disclose_fallback_noop_when_chunks_present(self):
        result = self._run([_make_chunk()], disclose_fallback=True)
        assert result.retrieval_metadata["grounded"] is True
        assert result.sources == ["Fizazi et al. 2017 NEJM (RCT, n=1199)"]


# ── _reclassify_intent language: custom system_prompt is authoritative ────────

class TestReclassifyIntentLanguage:
    """Regression coverage: intent-label language must follow whatever
    system_prompt is actually in effect, not just the `language` param —
    these can diverge when a caller overrides system_prompt without also
    updating `language` to match."""

    def _run_with_intent_response(self, intent_json: str, **generate_kwargs):
        llm = _make_llm(tool_return=_DEFAULT_TOOL_RETURN)
        llm.complete.return_value = MagicMock(content=intent_json)
        gen = CardGenerator(llm_client=llm)
        with (
            patch("src.generation.card_generator._load_withdrawals", return_value=()),
            patch("src.generation.card_generator._load_biomarker_entries", return_value=()),
        ):
            gen.generate_card(
                patient_id="P1",
                cancer_type="prostate",
                age_range="",
                clinical_history="mCRPC",
                comorbidities={},
                ranked_chunks=[_make_chunk()],
                confidence_score=0.8,
                **generate_kwargs,
            )
        # Inspect the actual system text sent to the intent-reclassification call.
        return llm.complete.call_args.args[0]

    def test_custom_french_prompt_wins_over_english_language_param(self):
        """language='en' (e.g. a UI default) but the caller's own system_prompt
        is French — intent labels must follow the French prompt, not 'en'."""
        system_sent = self._run_with_intent_response(
            '{"ADT + Abiraterone 1000 mg/day": "Palliatif"}',
            system_prompt="Répondez entièrement en français.",
            language="en",
        )
        assert "French intent labels only" in system_sent

    def test_custom_english_prompt_wins_over_french_default_language(self):
        """language defaults to 'fr' but the caller's own system_prompt is
        English — intent labels must follow the English prompt, not the default."""
        system_sent = self._run_with_intent_response(
            '{"ADT + Abiraterone 1000 mg/day": "Palliative"}',
            system_prompt="Answer entirely in English.",
        )
        assert "English intent labels only" in system_sent

    def test_no_override_falls_back_to_language_param(self):
        """No custom system_prompt at all — the default prompt was itself
        built from `language`, so `language` alone is authoritative."""
        system_sent = self._run_with_intent_response(
            '{"ADT + Abiraterone 1000 mg/day": "Palliative"}',
            language="en",
        )
        assert "English intent labels only" in system_sent
