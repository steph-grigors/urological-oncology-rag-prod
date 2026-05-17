"""
Treatment card generator for the /treatment-card endpoint.

Pipeline per request:
  1. translate_to_english()  — French clinical narrative → English retrieval query
  2. [retrieval happens in the route handler]
  3. generate_card()         — tool_use call → structured card dict
  4. _reclassify_intent()    — S1: second LLM call (stage + names only → intent labels)
  5. _strip_doc_tags()       — remove all [Doc N] from output text fields
  6. _apply_regulatory()     — French structured warnings from regulatory_withdrawals.json
  7. _apply_biomarker()      — French eligibility warnings from biomarker_eligibility.json
"""

from __future__ import annotations

import functools
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.generation.post_process import _load_withdrawals

if TYPE_CHECKING:
    from src.generation.llm_client import LLMClient
    from src.retrieval.reranker import RankedChunk

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DOC_TAG_RE = re.compile(r"\s*\[Doc \d+\]", re.IGNORECASE)


# ── Data loaders ──────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _load_biomarker_entries() -> tuple[dict, ...]:
    path = _DATA_DIR / "biomarker_eligibility.json"
    if not path.exists():
        logger.warning("biomarker_eligibility.json not found — biomarker checks skipped")
        return ()
    try:
        raw = json.loads(path.read_text())
        return tuple(raw.get("entries", []))
    except Exception:
        logger.exception("Failed to load biomarker_eligibility.json — checks skipped")
        return ()


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class TreatmentWarning:
    type: str         # "regulatory" | "biomarker"
    drug: str
    jurisdiction: str
    message: str


@dataclass
class TreatmentTriplet:
    drug: str
    intent: str       # Curatif | Palliatif | Adjuvant
    level: str        # A | B | C | Avis d'expert
    warnings: list[TreatmentWarning] = field(default_factory=list)


@dataclass
class TreatmentCardResult:
    patient_id: str
    stage: str
    confidence: str
    guideline: str
    comorbidities_impact: str
    treatment: list[TreatmentTriplet]
    treatment_confidence: str
    sources: list[str]
    retrieval_metadata: dict
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


# ── Card generator ────────────────────────────────────────────────────────────

class CardGenerator:
    """
    Generates French structured treatment cards from patient data + ranked chunks.
    Structured output is guaranteed via tool_use (Anthropic) or JSON mode (OpenAI).
    """

    _CARD_TOOL: dict = {
        "name": "generate_treatment_card",
        "description": (
            "Generate a structured oncology treatment card from the provided clinical evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stage": {
                    "type": "string",
                    "description": "TNM staging, ISUP grade, key biomarkers (PSA, etc.).",
                },
                "confidence": {
                    "type": "string",
                    "description": (
                        "Overall evidence quality. Use the language specified in the system prompt "
                        "(e.g. 'High' / 'Moderate' / 'Insufficient' in English, "
                        "'Élevée' / 'Modérée' / 'Insuffisante' in French)."
                    ),
                },
                "guideline": {
                    "type": "string",
                    "description": "Reference guideline (e.g. EAU 2024, NONE).",
                },
                "comorbidities_impact": {
                    "type": "string",
                    "description": "Impact of comorbidities on treatment selection.",
                },
                "treatment": {
                    "type": "array",
                    "description": "Recommended treatment options.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "drug": {
                                "type": "string",
                                "description": "Drug/regimen name and dosage if known.",
                            },
                            "intent": {
                                "type": "string",
                                "description": (
                                    "Therapeutic intent. Use the language specified in the system "
                                    "prompt (e.g. 'Curative' / 'Palliative' / 'Adjuvant' in English, "
                                    "'Curatif' / 'Palliatif' / 'Adjuvant' in French)."
                                ),
                            },
                            "level": {
                                "type": "string",
                                "enum": ["A", "B", "C", "Expert opinion"],
                                "description": "Evidence level.",
                            },
                        },
                        "required": ["drug", "intent", "level"],
                    },
                },
                "treatment_confidence": {
                    "type": "string",
                    "description": "Confidence in the treatment recommendations. Same language as 'confidence'.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Formatted references: 'Author et al. Year Journal (design, n=N)'. "
                        "Do NOT include [Doc N] tags."
                    ),
                },
            },
            "required": [
                "stage", "confidence", "guideline", "comorbidities_impact",
                "treatment", "treatment_confidence", "sources",
            ],
        },
    }

    # Default system prompt — English. Override via system_prompt arg for other languages.
    _CARD_SYSTEM = (
        "You are a clinical oncologist specializing in urological oncology. "
        "You generate structured treatment cards for qualified healthcare professionals, "
        "based exclusively on the provided scientific evidence.\n\n"
        "RULES:\n"
        "- All output in English.\n"
        "- Do NOT include [Doc N] tags in any output fields.\n"
        "- Format sources as: 'Author et al. Year Journal (design, n=N)'.\n"
        "- Base recommendations only on the provided documents. "
        "If evidence is insufficient, set confidence='Insufficient'.\n"
        "- Medical content must be precise, concise, and appropriate for clinical use."
    )

    _TRANSLATE_SYSTEM = (
        "You are a medical translator specializing in oncology. "
        "Translate the following clinical case description to a concise English literature "
        "search query (1-2 sentences). Focus on: cancer type, stage, treatment line, "
        "key biomarkers. Output only the English query."
    )

    _INTENT_SYSTEM = (
        "You are a clinical oncologist. Classify the therapeutic intent of each treatment "
        "based only on the staging information provided.\n\n"
        "Rules:\n"
        "- M0 / localized / locally advanced → Curative or Adjuvant\n"
        "- M1 / castration-resistant / metastatic → Palliative\n"
        "- Adjuvant: administered after curative surgery/radiotherapy to reduce recurrence\n\n"
        "Use the same language as the treatment card (English by default; "
        "French if the system prompt specified French output).\n\n"
        "Output valid JSON only, no explanation: {\"treatment_name\": \"intent\", ...}"
    )

    def __init__(self, llm_client: "LLMClient") -> None:
        self._llm = llm_client

    # ── Public API ────────────────────────────────────────────────────────

    def translate_to_english(self, french_text: str) -> str:
        """Translate French clinical narrative to an English retrieval query."""
        try:
            resp = self._llm.complete(
                self._TRANSLATE_SYSTEM,
                [{"role": "user", "content": french_text}],
                max_tokens=200,
            )
            return resp.content.strip()
        except Exception as exc:
            logger.warning("Query translation failed, using original text: %s", exc)
            return french_text

    def generate_card(
        self,
        patient_id: str,
        cancer_type: str,
        age_range: str,
        clinical_history: str,
        comorbidities: dict,
        ranked_chunks: list["RankedChunk"],
        confidence_score: float,
        corpus_version: str = "",
        system_prompt: str | None = None,
    ) -> TreatmentCardResult:
        """Full card generation pipeline from patient data + retrieved chunks."""
        t_start = time.monotonic()
        total_prompt = 0
        total_completion = 0
        active_system = system_prompt if system_prompt is not None else self._CARD_SYSTEM

        # ── Step 1: build the user prompt ─────────────────────────────────
        context_block = _build_context_block(ranked_chunks)
        comorbidities_str = _format_comorbidities(comorbidities)
        age_str = f" ({age_range})" if age_range else ""

        user_prompt = (
            f"Données patient:\n"
            f"- Type de cancer: {cancer_type}{age_str}\n"
            f"- Comorbidités: {comorbidities_str}\n\n"
            f"Anamnèse clinique:\n{clinical_history}\n\n"
            f"Documents de référence:\n{context_block}"
        )

        # ── Step 2: card generation via tool_use ──────────────────────────
        raw = self._llm.complete_with_tools(
            system=active_system,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[self._CARD_TOOL],
            max_tokens=2000,
        )
        if raw is None:
            logger.error("Card tool_use call returned no result")
            raw = {"input": {}, "prompt_tokens": 0, "completion_tokens": 0}

        card_data: dict = raw.get("input", {})
        total_prompt += raw.get("prompt_tokens", 0)
        total_completion += raw.get("completion_tokens", 0)

        # ── Step 3: S1 intent reclassification ────────────────────────────
        stage_raw = card_data.get("stage", "")
        treatment_list = card_data.get("treatment", [])
        treatment_names = [t.get("drug", "") for t in treatment_list if t.get("drug")]

        if stage_raw and treatment_names:
            intent_map = self._reclassify_intent(stage_raw, treatment_names)
            for t in treatment_list:
                drug = t.get("drug", "")
                for key, intent in intent_map.items():
                    if key.lower() in drug.lower() or drug.lower() in key.lower():
                        t["intent"] = intent
                        break

        # ── Step 4: build triplets (strip [Doc N] from drug names) ────────
        triplets = [
            TreatmentTriplet(
                drug=_strip_doc_tags(t.get("drug", "")),
                intent=t.get("intent", "Palliatif"),
                level=t.get("level", "B"),
            )
            for t in treatment_list
        ]

        # ── Step 5: strip [Doc N] from all text fields ────────────────────
        stage = _strip_doc_tags(stage_raw)
        guideline = _strip_doc_tags(card_data.get("guideline", ""))
        comorbidities_impact = _strip_doc_tags(card_data.get("comorbidities_impact", ""))
        sources = [_strip_doc_tags(s) for s in card_data.get("sources", [])]

        # ── Step 6: regulatory withdrawal warnings (French) ───────────────
        patient_context = f"{cancer_type} {clinical_history}".lower()
        _apply_regulatory_to_triplets(triplets, patient_context)

        # ── Step 7: biomarker eligibility warnings ────────────────────────
        _apply_biomarker_to_triplets(triplets, patient_context)

        latency_ms = (time.monotonic() - t_start) * 1000
        return TreatmentCardResult(
            patient_id=patient_id,
            stage=stage,
            confidence=card_data.get("confidence", "Modérée"),
            guideline=guideline,
            comorbidities_impact=comorbidities_impact,
            treatment=triplets,
            treatment_confidence=card_data.get("treatment_confidence", "Modérée"),
            sources=sources,
            retrieval_metadata={
                "chunks_used": len(ranked_chunks),
                "confidence_score": round(confidence_score, 4),
                "corpus_version": corpus_version,
            },
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            latency_ms=latency_ms,
        )

    # ── Private ───────────────────────────────────────────────────────────

    def _reclassify_intent(
        self,
        stage: str,
        treatment_names: list[str],
    ) -> dict[str, str]:
        """S1: intent-only LLM call using staging rules, not retrieval framing."""
        names_block = "\n".join(f"- {n}" for n in treatment_names)
        prompt = f"Stage: {stage}\n\nTreatments:\n{names_block}"
        try:
            resp = self._llm.complete(
                self._INTENT_SYSTEM,
                [{"role": "user", "content": prompt}],
                max_tokens=400,
            )
            raw = resp.content.strip()
            if "```" in raw:
                raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Intent reclassification failed: %s", exc)
            return {}


# ── Module-level helpers ──────────────────────────────────────────────────────

def _strip_doc_tags(text: str) -> str:
    return _DOC_TAG_RE.sub("", text).strip()


def _build_context_block(chunks: list["RankedChunk"], max_chars: int = 8000) -> str:
    """Format ranked chunks as numbered [Doc N] blocks for the card generation prompt."""
    parts: list[str] = []
    total = 0
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata if hasattr(chunk, "metadata") else {}
        title = meta.get("title", "Unknown")
        year = meta.get("year", "")
        design = meta.get("study_design", "")
        n = meta.get("sample_size")

        header = f"[Doc {i}] {title}"
        if year:
            header += f" ({year})"
        if design:
            header += f" — {design}"
        if n:
            header += f" (n={n})"

        text = chunk.text if hasattr(chunk, "text") else ""
        block = f"{header}\n{text}"

        if total + len(block) + 2 > max_chars:
            remaining = max_chars - total - 2
            if remaining > 100:
                parts.append(block[:remaining] + "…")
            break

        parts.append(block)
        total += len(block) + 2

    return "\n\n".join(parts)


def _format_comorbidities(comorbidities: dict) -> str:
    if not comorbidities:
        return "Aucune comorbidité précisée"
    return ", ".join(
        f"{k}: {v}" for k, v in comorbidities.items()
    )


def _apply_regulatory_to_triplets(
    triplets: list[TreatmentTriplet],
    patient_context: str,
) -> None:
    """
    Add French regulatory withdrawal warnings to matching triplets.
    Fires when drug name matches AND indication keyword found in patient context.
    """
    entries = _load_withdrawals()
    if not entries:
        return

    for triplet in triplets:
        drug_lower = triplet.drug.lower()
        for entry in entries:
            names = [entry.get("drug", "")] + entry.get("aliases", [])
            if not any(n.lower() in drug_lower for n in names if n):
                continue
            indication_kws = entry.get("indication_keywords", [])
            if indication_kws and not any(
                kw.lower() in patient_context for kw in indication_kws
            ):
                continue
            message = entry.get("warning_fr") or entry.get("warning", "")
            if message:
                triplet.warnings.append(
                    TreatmentWarning(
                        type="regulatory",
                        drug=entry.get("drug", ""),
                        jurisdiction=entry.get("jurisdiction", ""),
                        message=message,
                    )
                )


def _apply_biomarker_to_triplets(
    triplets: list[TreatmentTriplet],
    patient_context: str,
) -> None:
    """
    Add French biomarker eligibility warnings when required biomarker is absent
    from the patient context.
    """
    entries = _load_biomarker_entries()
    if not entries:
        return

    for triplet in triplets:
        drug_lower = triplet.drug.lower()
        for entry in entries:
            patterns = entry.get("drug_patterns", [])
            if not any(p.lower() in drug_lower for p in patterns):
                continue
            biomarker_kws = entry.get("biomarker_keywords", [])
            if any(kw.lower() in patient_context for kw in biomarker_kws):
                continue  # biomarker IS documented — no warning needed
            message = entry.get("warning_fr", "")
            if message:
                triplet.warnings.append(
                    TreatmentWarning(
                        type="biomarker",
                        drug=triplet.drug,
                        jurisdiction="biomarker",
                        message=message,
                    )
                )
