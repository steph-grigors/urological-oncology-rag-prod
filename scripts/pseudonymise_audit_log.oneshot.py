"""
One-shot: pseudonymise identifiers already written to audit_log.

Two things were persisted in the clear before the fixes in
fix/api-observability-hygiene and fix/audit-pii:

  user_id     the raw API key, on every authenticated row
  question    "[treatment-card] patient=<id>: <narrative>" for card rows

Both are rewritten in place to the same stable tokens the application now
writes, so rows from before and after the change are consistent and a
reviewer can still group by caller and by patient.

Deliberately NOT done here: the clinical narrative is left as it is. That was
an explicit decision -- pseudonymise the identifier only. Be clear about what
that does and does not achieve: a history detailed enough to warrant a
treatment card can identify a person without any identifier attached, so
these rows are pseudonymised, not anonymous.

Run with --dry-run first. It reports exactly what would change and touches
nothing.

    python scripts/pseudonymise_audit_log.oneshot.py --dry-run
    python scripts/pseudonymise_audit_log.oneshot.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, text

from src.api.middleware.auth import api_key_fingerprint
from src.observability.scrub import pseudonymise_patient_id

_CARD_PREFIX_RE = re.compile(r"^\[treatment-card\] patient=([^:]*):")


def _engine(db_url: str):
    return create_engine(db_url.replace("postgresql+asyncpg://", "postgresql://"))


def run(db_url: str, dry_run: bool) -> int:
    engine = _engine(db_url)
    changed_keys = changed_patients = 0

    with engine.begin() as conn:
        rows = conn.execute(text(
            "select query_id, user_id, question from audit_log"
        )).fetchall()

        for query_id, user_id, question in rows:
            updates = {}

            # Raw API key -> the same fingerprint the app writes now.
            if user_id and not user_id.startswith("k_"):
                updates["user_id"] = api_key_fingerprint(user_id)
                changed_keys += 1

            # patient=<id> -> stable token. Already-tokenised rows are skipped.
            if question:
                m = _CARD_PREFIX_RE.match(question)
                if m and not m.group(1).startswith("p_"):
                    token = pseudonymise_patient_id(m.group(1))
                    updates["question"] = (
                        f"[treatment-card] patient={token}:" + question[m.end():]
                    )
                    changed_patients += 1

            if updates and not dry_run:
                sets = ", ".join(f"{k} = :{k}" for k in updates)
                conn.execute(
                    text(f"update audit_log set {sets} where query_id = :qid"),
                    {**updates, "qid": query_id},
                )

        print(f"  rows examined            : {len(rows)}")
        print(f"  raw API keys {'to rewrite' if dry_run else 'rewritten '}  : {changed_keys}")
        print(f"  patient ids  {'to rewrite' if dry_run else 'rewritten '}  : {changed_patients}")
        if dry_run:
            print("  DRY RUN — nothing was written")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db-url", default=os.environ.get("POSTGRES_URL", ""))
    args = ap.parse_args()
    if not args.db_url:
        raise SystemExit("POSTGRES_URL not set and --db-url not given")
    raise SystemExit(run(args.db_url, args.dry_run))
