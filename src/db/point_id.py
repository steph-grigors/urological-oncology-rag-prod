"""
The single definition of a chunk's Qdrant point id.

Two different derivations existed, and they disagreed:

    src/db/vector_store.py   uuid5(NAMESPACE_DNS, chunk_id)
    src/ingestion/embed.py   UUID(bytes=md5(chunk_id))

The same chunk_id therefore produced two different point ids depending on
which path wrote it. Since a Qdrant upsert replaces by id, a chunk written
through one path and later re-written through the other would be stored twice
rather than updated, silently doubling that chunk in retrieval.

Production's 713,883 points were all written through the ingestion path, so
that derivation is the one kept. The uuid5 variant was only ever exercised by
tests.

On MD5: this is a content address, not a security primitive. It maps a stable
string to a stable identifier, and nothing depends on it being hard to find a
collision for. Changing it to SHA-256 would be cosmetically tidier and would
orphan every existing point, requiring a full re-embed of the corpus. If a
scanner flags this line, that is the reason it stays.

This module deliberately imports nothing beyond the standard library, so both
the ingestion path and the db layer can use it without dragging qdrant_client
into contexts that do not need it.
"""

from __future__ import annotations

import hashlib
import uuid

__all__ = ["chunk_point_id"]


def chunk_point_id(chunk_id: str) -> str:
    """Return the deterministic Qdrant point id for `chunk_id`.

    Stable across processes and runs: the same chunk_id always yields the same
    id, which is what makes re-ingesting a paper update its chunks in place
    rather than duplicating them.
    """
    return str(uuid.UUID(bytes=hashlib.md5(chunk_id.encode()).digest()))
