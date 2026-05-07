"""
All prompt templates for the generation layer.
"""

from __future__ import annotations

from config.constants import MEDICAL_DISCLAIMER

SYSTEM_PROMPT = (
    "You are a clinical evidence summarization assistant specializing in urological oncology.\n\n"
    "Your role is to synthesize evidence from peer-reviewed literature for qualified healthcare "
    "professionals. You do not provide personalized medical advice, diagnoses, or treatment decisions.\n\n"
    "SCOPE: Only answer questions about urological oncology (prostate, bladder, kidney, "
    "testicular cancer). If a question is outside this scope, say so clearly.\n\n"
    "CITATION RULES:\n"
    "- Every factual claim must be supported by an inline [Doc N] citation.\n"
    "- Use only the documents provided — never fabricate or infer sources.\n"
    "- If the context does not contain sufficient information, state that explicitly.\n\n"
    "TEMPORAL CONFLICT RULES (critical for clinical safety):\n"
    "- Each document header includes its publication year. Always consider it.\n"
    "- When sources span more than 5 years, note this explicitly in the summary.\n"
    "- When a newer source reaches a different conclusion than an older one on the same "
    "intervention or outcome, you MUST flag this: state which is newer, what each concluded, "
    "and that the newer evidence should be weighted more heavily pending independent review.\n"
    "- Never present findings from an older study as current consensus if a newer study "
    "in the same context contradicts or supersedes it.\n"
    "- If all sources are older than 5 years, note that guidelines may have since been updated.\n\n"
    "OUTPUT FORMAT (always use these four sections):\n\n"
    "## CLINICAL EVIDENCE SUMMARY\n"
    "[Main answer with [Doc N] inline citations for each factual claim]\n\n"
    "## EVIDENCE QUALITY\n"
    "[Brief assessment: study designs represented, sample sizes, consistency of findings, "
    "and date range of sources — flag if the evidence base is more than 5 years old]\n\n"
    "## SOURCES\n"
    "[Doc 1]: <title> (<year>) — <key finding or relevance>\n"
    "[Doc 2]: ...\n\n"
    "## LIMITATIONS\n"
    "[Evidence gaps, conflicting results, temporal conflicts between sources, "
    "generalisability concerns, or reasons for caution]"
    + MEDICAL_DISCLAIMER
)

USER_PROMPT_TEMPLATE = """{context_block}

---

**Clinical question:** {question}

Please summarise the evidence above to answer this question. Cite each document you use as [Doc N].
"""

HEDGED_ANSWER_PREFIX = (
    "**Note:** The available evidence for this question is limited or of lower quality. "
    "Interpret the following summary with caution and consider consulting primary literature "
    "or current clinical guidelines.\n\n"
)

LOW_CONFIDENCE_REFUSAL = (
    "I cannot provide a reliable evidence summary for this question. "
    "The retrieved literature does not contain sufficient relevant information to answer confidently. "
    "Please consult current clinical guidelines (e.g. EAU, NCCN), recent systematic reviews, "
    "or a specialist in urological oncology."
)

FALLBACK_DISCLAIMER = (
    "> ⚠️ **Knowledge base disclaimer**: No sufficiently relevant literature was found "
    "in the knowledge base for this query. The following response draws on the model's "
    "general medical knowledge and has **not** been verified against peer-reviewed sources "
    "in this database. Treat with appropriate caution and verify against current guidelines.\n\n"
)

FALLBACK_USER_TEMPLATE = (
    "**Clinical question:** {question}\n\n"
    "Note: The knowledge base did not return sufficiently relevant literature for this query. "
    "Answer based on your general medical knowledge and training. "
    "Clearly indicate where you are drawing on established guidelines versus general knowledge, "
    "and flag any areas of uncertainty."
)

QUERY_REWRITE_PROMPT = (
    "Given the conversation history below and a follow-up question, rewrite the follow-up as a "
    "fully self-contained standalone question that captures all necessary context from the history.\n\n"
    "CONVERSATION HISTORY:\n"
    "{history}\n\n"
    "FOLLOW-UP QUESTION: {question}\n\n"
    "STANDALONE QUESTION:"
)


def format_context_block(chunks: list, max_chars: int = 8000) -> str:
    """Format a list of chunks into numbered [Doc N] blocks, truncated to max_chars."""
    blocks: list[str] = []
    total = 0

    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata if hasattr(chunk, "metadata") else {}
        title = meta.get("title", "Unknown title")
        year = meta.get("year", "")
        section = meta.get("section", "")
        design = meta.get("study_design", "")

        header_parts = [f"[Doc {i}]", title]
        if year:
            header_parts.append(str(year))
        if section:
            header_parts.append(section)
        if design:
            header_parts.append(design)

        text = chunk.text if hasattr(chunk, "text") else str(chunk)
        block = " | ".join(header_parts) + "\n" + text

        if total + len(block) + 2 > max_chars:
            remaining = max_chars - total - 2
            if remaining > 100:
                blocks.append(block[:remaining] + "…")
            break

        blocks.append(block)
        total += len(block) + 2  # +2 for the "\n\n" separator

    return "\n\n".join(blocks)


def build_prompt(
    question: str,
    chunks: list,
    max_context_chars: int = 8000,
    confidence_level: str = "high",
) -> list[dict]:
    """Return the user-turn messages list ready for LLMClient.complete()."""
    context = format_context_block(chunks, max_context_chars)
    user_content = USER_PROMPT_TEMPLATE.format(context_block=context, question=question)
    if confidence_level != "high":
        user_content = HEDGED_ANSWER_PREFIX + user_content
    return [{"role": "user", "content": user_content}]
