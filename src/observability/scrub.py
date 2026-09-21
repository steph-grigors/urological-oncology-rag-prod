"""
De-identification helpers applied on the way into the audit log.

The audit log is the one place where free-text clinical input is persisted.
`config.constants.PII_SCRUB_PATTERNS` has always described what should be
removed from it and was referenced from nowhere, so every question and every
patient narrative was written verbatim.

Two distinct jobs, deliberately kept separate:

  pseudonymise_patient_id  replaces a caller-supplied patient identifier with
                           a stable non-reversible token. The same identifier
                           always maps to the same token, so a reviewer can
                           still see that several cards concern one patient,
                           follow a case across rows, and count distinct
                           patients -- without the identifier itself being in
                           the table.

  scrub_pii                removes structured identifiers that appear inside
                           free text: national insurance numbers, long digit
                           runs that look like medical record numbers, and
                           email addresses.

Neither makes a clinical narrative anonymous. A history detailed enough to be
worth a treatment card can identify someone on its own, and no regex changes
that. These reduce exposure; they are not a substitute for a decision about
whether the narrative should be retained at all.
"""

from __future__ import annotations

import hashlib
import re
from typing import Final

from config.constants import PII_SCRUB_PATTERNS

_COMPILED: Final = [re.compile(p) for p in PII_SCRUB_PATTERNS]
_REDACTED: Final = "[redacted]"

_PATIENT_TOKEN_PREFIX: Final = "p_"
_TOKEN_LENGTH: Final = 16


def pseudonymise_patient_id(patient_id: str) -> str:
    """Return a stable, non-reversible token for a patient identifier.

    Empty input returns empty, so a caller that sent no identifier is not
    given a token that implies one existed.
    """
    if not patient_id:
        return ""
    digest = hashlib.sha256(patient_id.encode("utf-8")).hexdigest()
    return f"{_PATIENT_TOKEN_PREFIX}{digest[:_TOKEN_LENGTH]}"


def scrub_pii(text: str) -> str:
    """Replace structured identifiers in free text with a redaction marker."""
    if not text:
        return text
    for pattern in _COMPILED:
        text = pattern.sub(_REDACTED, text)
    return text


def audit_question_for_card(patient_id: str, clinical_history: str, limit: int = 500) -> str:
    """Build the `question` value stored for a treatment card.

    Keeps the existing shape -- "[treatment-card] patient=X: <history>" -- so
    rows written before and after this change stay greppable the same way,
    with the identifier pseudonymised and the narrative scrubbed.
    """
    token = pseudonymise_patient_id(patient_id)
    return f"[treatment-card] patient={token}: {scrub_pii(clinical_history)[:limit]}"
