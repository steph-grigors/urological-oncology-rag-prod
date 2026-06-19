"""
Treatment card generator for the /treatment-card endpoint.

Pipeline per request:
  1. translate_to_english()  — clinical narrative (any language) → English retrieval query
  2. [retrieval happens in the route handler]
  3. generate_card()         — tool_use call → structured card dict
  4. _reclassify_intent()    — S1: second LLM call (stage + names only → intent labels)
  5. citation handling       — see below
  6. _apply_regulatory()     — structured warnings from regulatory_withdrawals.json
  7. _apply_biomarker()      — eligibility warnings from biomarker_eligibility.json

Language:
  `generate_card(..., language="fr"|"en")` controls everything the server itself
  injects (user-prompt scaffold labels, silent fallback values, intent-language
  instruction, and which translation of regulatory/biomarker warnings is picked).
  Default is "fr" to preserve the exact behaviour existing callers (e.g. the
  onco-review-app notebook, which always supplies its own French system_prompt)
  already depend on. It does NOT change the `treatment[].level` enum, which is
  always one of the language-neutral codes "A"/"B"/"C"/"Expert opinion" enforced
  by the tool schema.

Citations (`keep_citations`, default False):
  By default every field is stripped of `[Doc N]` tags (today's behaviour, kept
  for callers like onco-review-app that never set this). When True:
    - `treatment[].drug` may keep a `[Doc N]` tag, but any tag whose N doesn't
      correspond to an actually-retrieved chunk is removed (range-validated,
      same principle as /query's hallucinated-citation check). The drug/dosage
      text itself is still LLM-authored — this guarantees the *citation pointer*
      is always valid, not that the clinical recommendation is correct.
    - `sources` is NOT trusted as LLM-authored prose at all: the LLM only signals
      *which* `[Doc N]` it used, and the actual citation text is regenerated
      from real chunk metadata (same data `sources_detail` uses) — this field is
      hallucination-free by construction, not just hallucination-checked.

Fallback disclosure (`disclose_fallback`, default False):
  When True and no chunks were retrieved (parametric-knowledge fallback), both
  `sources` and `sources_detail` are replaced with an explicit disclosure entry
  and `retrieval_metadata["grounded"] = False` is set. Default False so this
  doesn't change onco-review-app's `retrieval_metadata` shape.
"""

from __future__ import annotations

import functools
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.generation.post_process import _load_withdrawals

if TYPE_CHECKING:
    from src.generation.llm_client import LLMClient
    from src.retrieval.reranker import RankedChunk

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DOC_TAG_RE = re.compile(r"\s*\[Doc \d+\]", re.IGNORECASE)
_DOC_TAG_CAPTURE_RE = re.compile(r"\s*\[Doc (\d+)\]", re.IGNORECASE)

CardLanguage = Literal["fr", "en"]

_FALLBACK_DISCLOSURE: dict[str, str] = {
    "fr": (
        "⚠ Aucune littérature pertinente n'a été retrouvée — cette recommandation "
        "s'appuie sur les connaissances générales du modèle, et non sur la base "
        "documentaire indexée. À vérifier impérativement avant usage clinique."
    ),
    "en": (
        "⚠ No relevant literature was retrieved — this recommendation is based on "
        "the model's general clinical knowledge, not the indexed evidence base. "
        "Must be verified before clinical use."
    ),
}

# Labels and silent-fallback values the server injects itself (as opposed to
# text the LLM generates). Keyed by language; "fr" is the long-standing default
# so existing callers that never set `language` see byte-identical output.
_LABELS: dict[str, dict[str, str]] = {
    "fr": {
        "patient_data": "Données patient",
        "cancer_type": "Type de cancer",
        "comorbidities": "Comorbidités",
        "clinical_history": "Anamnèse clinique",
        "reference_docs": "Documents de référence",
        "no_comorbidities": "Aucune comorbidité précisée",
        "default_confidence": "Modérée",
        "default_intent": "Palliatif",
    },
    "en": {
        "patient_data": "Patient data",
        "cancer_type": "Cancer type",
        "comorbidities": "Comorbidities",
        "clinical_history": "Clinical history",
        "reference_docs": "Reference documents",
        "no_comorbidities": "No comorbidities specified",
        "default_confidence": "Moderate",
        "default_intent": "Palliative",
    },
}


def _labels(language: str) -> dict[str, str]:
    return _LABELS.get(language, _LABELS["fr"])


def _localized_warning(entry: dict, language: str) -> str:
    """Pick the warning text for `language`, falling back to the other
    language rather than silently dropping a clinical-safety warning."""
    if language == "fr":
        return entry.get("warning_fr") or entry.get("warning", "")
    return entry.get("warning") or entry.get("warning_fr", "")


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


# ── Tool schema + default system prompt builders ────────────────────────────
# Both depend on `keep_citations`/`language`, so they're functions of those
# params rather than static class constants — but there are only 2 and 4
# possible outputs respectively, so each is cached after first build instead
# of re-doing the dict/string construction on every request. Callers must
# treat the returned dict as read-only (it's shared across calls via the cache).

@functools.lru_cache(maxsize=2)
def _build_card_tool(keep_citations: bool) -> dict:
    drug_description = "Drug/regimen name and dosage if known."
    sources_description = (
        "Formatted references: 'Author et al. Year Journal (design, n=N)'. "
        "Do NOT include [Doc N] tags."
    )
    if keep_citations:
        drug_description += (
            " If supported by one of the provided documents, append its citation "
            "marker, e.g. 'Abiraterone 1000 mg/day [Doc 2]'. Only use a [Doc N] "
            "you were actually given — never invent one."
        )
        sources_description = (
            "For each provided document that supports a treatment recommendation, "
            "list ONLY its citation marker, e.g. '[Doc 2]'. One entry per document "
            "used; omit documents you didn't rely on. The full citation text is "
            "generated automatically — do not write author/journal/year yourself."
        )

    return {
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
                                "description": drug_description,
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
                    "description": sources_description,
                },
            },
            "required": [
                "stage", "confidence", "guideline", "comorbidities_impact",
                "treatment", "treatment_confidence", "sources",
            ],
        },
    }


@functools.lru_cache(maxsize=4)
def _default_card_system(language: CardLanguage, keep_citations: bool) -> str:
    """Default system prompt, used only when the caller doesn't pass its own
    `system_prompt`. "fr" + keep_citations=False is the long-standing default."""
    if language == "fr":
        rules = "- Toute la sortie doit être en français.\n"
        if keep_citations:
            rules += (
                "- Pour le champ 'treatment', ajoutez une balise [Doc N] après le "
                "traitement lorsqu'il est soutenu par un document fourni.\n"
                "- Pour le champ 'sources', indiquez uniquement la balise [Doc N] de "
                "chaque document utilisé — le texte de citation sera généré "
                "automatiquement.\n"
                "- N'incluez de balises [Doc N] dans aucun autre champ (stage, "
                "guideline, comorbidities_impact).\n"
            )
        else:
            rules += "- Ne pas inclure de balises [Doc N] dans les champs de sortie.\n"
        return (
            "Vous êtes un oncologue spécialisé en oncologie urologique. "
            "Vous générez des fiches de traitement structurées pour des professionnels "
            "de santé qualifiés, en vous basant exclusivement sur les preuves "
            "scientifiques fournies.\n\n"
            "RÈGLES :\n"
            f"{rules}"
            "- Formater les sources ainsi : « Auteur et al. Année Journal (design, n=N) ».\n"
            "- Baser les recommandations exclusivement sur les documents fournis. "
            "Si les preuves sont insuffisantes, indiquer confidence='Insuffisante'.\n"
            "- Le contenu médical doit être précis, concis et adapté à un usage clinique."
        )

    rules = "- All output in English.\n"
    if keep_citations:
        rules += (
            "- For the 'treatment' field, append a [Doc N] tag after the treatment "
            "when it is supported by a provided document.\n"
            "- For the 'sources' field, list only the [Doc N] tag of each document "
            "used — the citation text is generated automatically.\n"
            "- Do NOT include [Doc N] tags in any other field (stage, guideline, "
            "comorbidities_impact).\n"
        )
    else:
        rules += "- Do NOT include [Doc N] tags in any output fields.\n"
    return (
        "You are a clinical oncologist specializing in urological oncology. "
        "You generate structured treatment cards for qualified healthcare professionals, "
        "based exclusively on the provided scientific evidence.\n\n"
        "RULES:\n"
        f"{rules}"
        "- Format sources as: 'Author et al. Year Journal (design, n=N)'.\n"
        "- Base recommendations only on the provided documents. "
        "If evidence is insufficient, set confidence='Insufficient'.\n"
        "- Medical content must be precise, concise, and appropriate for clinical use."
    )


# ── Card generator ────────────────────────────────────────────────────────────

class CardGenerator:
    """
    Generates French structured treatment cards from patient data + ranked chunks.
    Structured output is guaranteed via tool_use (Anthropic) or JSON mode (OpenAI).
    """

    _TRANSLATE_SYSTEM = (
        "You are a medical translator specializing in oncology. "
        "Translate the following clinical case description (it may be in any language) to a "
        "concise English literature search query (1-2 sentences). Focus on: cancer type, "
        "stage, treatment line, key biomarkers. Output only the English query."
    )

    _INTENT_SYSTEM = (
        "You are a clinical oncologist. Classify the therapeutic intent of each treatment "
        "based only on the staging information provided.\n\n"
        "Rules:\n"
        "- M0 / localized / locally advanced → Curative or Adjuvant\n"
        "- M1 / castration-resistant / metastatic → Palliative\n"
        "- Adjuvant: administered after curative surgery/radiotherapy to reduce recurrence\n\n"
        "Output valid JSON only, no explanation: {\"treatment_name\": \"intent\", ...}"
    )

    def __init__(self, llm_client: "LLMClient") -> None:
        self._llm = llm_client

    # ── Properties (for audit logging) ──────────────────────────────────────

    @property
    def model(self) -> str:
        return self._llm.model

    @property
    def provider(self) -> str:
        return self._llm.provider

    # ── Public API ────────────────────────────────────────────────────────

    def translate_to_english(self, clinical_text: str) -> str:
        """Translate a clinical narrative (any language) to an English retrieval query."""
        try:
            resp = self._llm.complete(
                self._TRANSLATE_SYSTEM,
                [{"role": "user", "content": clinical_text}],
                max_tokens=200,
            )
            return resp.content.strip()
        except Exception as exc:
            logger.warning("Query translation failed, using original text: %s", exc)
            return clinical_text

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
        language: CardLanguage = "fr",
        keep_citations: bool = False,
        disclose_fallback: bool = False,
    ) -> TreatmentCardResult:
        """Full card generation pipeline from patient data + retrieved chunks.

        `language` controls server-injected text only (prompt scaffold labels,
        silent fallback values, intent-language instruction, and warning-text
        selection) — it does not affect what the LLM itself writes, which is
        governed entirely by `system_prompt` when one is supplied. Default
        "fr" preserves existing behaviour for callers that don't set it.

        `keep_citations` (default False, preserves existing behaviour) lets
        `[Doc N]` survive in `treatment[].drug` (range-validated only) and
        `sources` (fully regenerated from chunk metadata — see module docstring).

        `disclose_fallback` (default False) replaces `sources`/`sources_detail`
        with an explicit disclosure when no chunks were retrieved, instead of
        silently leaving them as whatever the LLM wrote from memory.
        """
        t_start = time.monotonic()
        total_prompt = 0
        total_completion = 0
        labels = _labels(language)
        default_system = _default_card_system(language, keep_citations)
        active_system = system_prompt if system_prompt is not None else default_system

        # ── Step 1: build the user prompt ─────────────────────────────────
        context_block = _build_context_block(ranked_chunks)
        comorbidities_str = _format_comorbidities(comorbidities, labels["no_comorbidities"])
        age_str = f" ({age_range})" if age_range else ""

        user_prompt = (
            f"{labels['patient_data']}:\n"
            f"- {labels['cancer_type']}: {cancer_type}{age_str}\n"
            f"- {labels['comorbidities']}: {comorbidities_str}\n\n"
            f"{labels['clinical_history']}:\n{clinical_history}\n\n"
            f"{labels['reference_docs']}:\n{context_block}"
        )

        # ── Step 2: card generation via tool_use ──────────────────────────
        raw = self._llm.complete_with_tools(
            system=active_system,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[_build_card_tool(keep_citations)],
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
            intent_map = self._reclassify_intent(
                stage_raw, treatment_names, language, custom_system_prompt=system_prompt
            )
            for t in treatment_list:
                drug = t.get("drug", "")
                for key, intent in intent_map.items():
                    if key.lower() in drug.lower() or drug.lower() in key.lower():
                        t["intent"] = intent
                        break

        # ── Step 4: build triplets ─────────────────────────────────────────
        # keep_citations=False (default): strip every [Doc N], as always.
        # keep_citations=True: keep [Doc N] only if it points at a real chunk.
        n_chunks = len(ranked_chunks)
        triplets = [
            TreatmentTriplet(
                drug=(
                    _strip_invalid_doc_tags(t.get("drug", ""), n_chunks)
                    if keep_citations else _strip_doc_tags(t.get("drug", ""))
                ),
                intent=t.get("intent", labels["default_intent"]),
                level=t.get("level", "B"),
            )
            for t in treatment_list
        ]

        # ── Step 5: stage/guideline/comorbidities_impact always stripped ──
        # No instruction ever asks the LLM to cite here, regardless of
        # keep_citations — this is a safety net, not a behavioural switch.
        stage = _strip_doc_tags(stage_raw)
        guideline = _strip_doc_tags(card_data.get("guideline", ""))
        comorbidities_impact = _strip_doc_tags(card_data.get("comorbidities_impact", ""))

        # `sources`: default behaviour unchanged. With keep_citations=True the
        # LLM's free text is discarded entirely and replaced with text rendered
        # from real chunk metadata for every [Doc N] it referenced — hallucination
        # is structurally impossible here, not just checked after the fact.
        sources_raw = card_data.get("sources", [])
        if isinstance(sources_raw, str):
            sources_raw = [sources_raw]
        if keep_citations:
            sources = _ground_sources(sources_raw, ranked_chunks)
        else:
            sources = [_strip_doc_tags(s) for s in sources_raw]

        # ── Step 6: regulatory withdrawal warnings ────────────────────────
        patient_context = f"{cancer_type} {clinical_history}".lower()
        _apply_regulatory_to_triplets(triplets, patient_context, language)

        # ── Step 7: biomarker eligibility warnings ────────────────────────
        _apply_biomarker_to_triplets(triplets, patient_context, language)

        # ── Step 8: grounded, database-derived source records ─────────────
        # Additive — does not replace the free-text `sources` field above.
        sources_detail = _build_sources_detail(ranked_chunks)

        retrieval_metadata = {
            "chunks_used": n_chunks,
            "confidence_score": round(confidence_score, 4),
            "corpus_version": corpus_version,
            "sources_detail": sources_detail,
        }

        # ── Step 9: fallback disclosure (opt-in, doesn't touch the shape of
        # retrieval_metadata for callers that never set disclose_fallback) ──
        if disclose_fallback:
            grounded = n_chunks > 0
            retrieval_metadata["grounded"] = grounded
            if not grounded:
                disclosure = _FALLBACK_DISCLOSURE.get(language, _FALLBACK_DISCLOSURE["fr"])
                sources = [disclosure]
                retrieval_metadata["sources_detail"] = [_disclosure_source_detail(disclosure)]

        latency_ms = (time.monotonic() - t_start) * 1000
        return TreatmentCardResult(
            patient_id=patient_id,
            stage=stage,
            confidence=card_data.get("confidence", labels["default_confidence"]),
            guideline=guideline,
            comorbidities_impact=comorbidities_impact,
            treatment=triplets,
            treatment_confidence=card_data.get("treatment_confidence", labels["default_confidence"]),
            sources=sources,
            retrieval_metadata=retrieval_metadata,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            latency_ms=latency_ms,
        )

    # ── Private ───────────────────────────────────────────────────────────

    def _reclassify_intent(
        self,
        stage: str,
        treatment_names: list[str],
        language: CardLanguage = "fr",
        custom_system_prompt: str | None = None,
    ) -> dict[str, str]:
        """S1: intent-only LLM call using staging rules, not retrieval framing.

        Language: when the caller supplied its own `system_prompt` (overriding
        the default), that text — not the `language` param — is authoritative,
        since the card body's actual language is whatever that prompt asked
        for, and `language` may not have been updated to match. When no
        override was supplied, the default prompt was itself built from
        `language`, so the two are always consistent and `language` is used
        directly.
        """
        names_block = "\n".join(f"- {n}" for n in treatment_names)
        prompt = f"Stage: {stage}\n\nTreatments:\n{names_block}"

        is_french = (
            "français" in custom_system_prompt.lower()
            if custom_system_prompt is not None
            else language == "fr"
        )

        system = self._INTENT_SYSTEM
        if is_french:
            system += "\nIMPORTANT: Use French intent labels only: Curatif / Palliatif / Adjuvant."
        else:
            system += "\nIMPORTANT: Use English intent labels only: Curative / Palliative / Adjuvant."

        try:
            resp = self._llm.complete(
                system,
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


def _strip_invalid_doc_tags(text: str, n_chunks: int) -> str:
    """Remove only the [Doc N] tags whose N doesn't point at a real chunk.
    Valid tags (1 <= N <= n_chunks) are left untouched."""
    def _replace(m: re.Match) -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= n_chunks else ""
    return _DOC_TAG_CAPTURE_RE.sub(_replace, text).strip()


def _ground_sources(sources_raw: list[str], ranked_chunks: list["RankedChunk"]) -> list[str]:
    """Discard the LLM's free-text source descriptions entirely; keep only
    which [Doc N] it referenced, and render the citation text ourselves from
    real chunk metadata. Hallucination-free by construction — there is no
    LLM-authored text left in the output of this function."""
    n_chunks = len(ranked_chunks)
    cited: list[int] = []
    seen: set[int] = set()
    for item in sources_raw:
        for m in _DOC_TAG_CAPTURE_RE.finditer(item):
            n = int(m.group(1))
            if 1 <= n <= n_chunks and n not in seen:
                seen.add(n)
                cited.append(n)
    return [_format_grounded_source(ranked_chunks[n - 1], n) for n in cited]


def _format_grounded_source(chunk: "RankedChunk", n: int) -> str:
    """Render one [Doc N] citation purely from chunk metadata — no LLM text."""
    meta = chunk.metadata if hasattr(chunk, "metadata") else {}
    authors_raw = meta.get("authors", [])
    first_author = ""
    if isinstance(authors_raw, list) and authors_raw:
        first_author = str(authors_raw[0]).split()[0] if authors_raw[0] else ""
    elif isinstance(authors_raw, str) and authors_raw:
        first_author = authors_raw.split()[0]
    authors_str = f"{first_author} et al." if first_author else ""

    year = meta.get("year")
    journal = meta.get("journal") or ""
    design = meta.get("study_design") or ""
    sample = meta.get("sample_size")

    header_parts = [p for p in (authors_str, str(year) if year else "", journal) if p]
    header = " ".join(header_parts) if header_parts else (meta.get("title") or "Unknown source")

    suffix_parts = [p for p in (design, f"n={sample}" if sample else "") if p]
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""

    return f"[Doc {n}] {header}{suffix}"


def _disclosure_source_detail(disclosure: str) -> dict:
    """sources_detail entry matching the SourceCard-like shape, for the
    parametric-knowledge-fallback disclosure (no real chunk backs it)."""
    return {
        "chunk_id": "",
        "title": disclosure,
        "authors": "",
        "journal": "",
        "year": None,
        "study_design": "parametric_knowledge",
        "sample_size": None,
        "section": "",
        "key_finding": "",
        "pmid": "",
    }


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


def _build_sources_detail(chunks: list["RankedChunk"]) -> list[dict]:
    """Grounded, database-derived source records — additive alongside the
    LLM-authored `sources` free-text field. Mirrors /query's SourceCard shape."""
    detail: list[dict] = []
    for chunk in chunks:
        meta = chunk.metadata if hasattr(chunk, "metadata") else {}
        authors_raw = meta.get("authors", [])
        if isinstance(authors_raw, list):
            authors_str = ", ".join(str(a) for a in authors_raw[:3])
            if len(authors_raw) > 3:
                authors_str += " et al."
        else:
            authors_str = str(authors_raw) if authors_raw else ""

        text = chunk.text if hasattr(chunk, "text") else ""
        detail.append({
            "chunk_id": getattr(chunk, "chunk_id", ""),
            "title": meta.get("title") or "Unknown",
            "authors": authors_str,
            "journal": meta.get("journal") or "",
            "year": meta.get("year"),
            "study_design": meta.get("study_design") or "",
            "sample_size": meta.get("sample_size"),
            "section": meta.get("section") or "",
            "key_finding": text[:150],
            "pmid": meta.get("pmid") or "",
        })
    return detail


def _format_comorbidities(comorbidities: dict, no_comorbidities_label: str = "Aucune comorbidité précisée") -> str:
    if not comorbidities:
        return no_comorbidities_label
    return ", ".join(
        f"{k}: {v}" for k, v in comorbidities.items()
    )


def _apply_regulatory_to_triplets(
    triplets: list[TreatmentTriplet],
    patient_context: str,
    language: CardLanguage = "fr",
) -> None:
    """
    Add regulatory withdrawal warnings to matching triplets.
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
            message = _localized_warning(entry, language)
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
    language: CardLanguage = "fr",
) -> None:
    """
    Add biomarker eligibility warnings when required biomarker is absent
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
            message = _localized_warning(entry, language)
            if message:
                triplet.warnings.append(
                    TreatmentWarning(
                        type="biomarker",
                        drug=triplet.drug,
                        jurisdiction="biomarker",
                        message=message,
                    )
                )
