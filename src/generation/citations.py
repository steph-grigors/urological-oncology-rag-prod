"""
Shared [Doc N]-style citation validation, used by both /query
(ClinicalGenerator._check_citations) and /treatment-card
(card_generator._strip_invalid_doc_tags).
"""

from __future__ import annotations

import re

DOC_TAG_RE = re.compile(r"\s*\[Doc (\d+)\]", re.IGNORECASE)


def strip_invalid_citations(text: str, num_docs: int) -> tuple[str, list[int]]:
    """Remove [Doc N] tags whose N doesn't point at a real chunk
    (1 <= N <= num_docs). Valid tags are left untouched.

    Returns (cleaned_text, sorted_unique_hallucinated_ns).
    """
    hallucinated: list[int] = []

    def _replace(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= num_docs:
            return m.group(0)
        if n not in hallucinated:
            hallucinated.append(n)
        return ""

    cleaned = DOC_TAG_RE.sub(_replace, text).strip()
    return cleaned, sorted(hallucinated)
