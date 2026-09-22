"""
Qdrant vector store abstraction.

Wraps qdrant-client with a domain-specific interface so no other module
imports from qdrant-client directly.  Collection uses a single default
(unnamed) dense vector — 1536-dim cosine (text-embedding-3-small) — matching
what src/ingestion/embed.py actually writes in production. Keyword search is
handled separately by the in-memory bm25s index (src/retrieval/bm25_search.py),
not by Qdrant.

Payload indexes are created on first call to ensure_collection so that
Qdrant can accelerate filtered searches without scanning all points.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from qdrant_client import QdrantClient
from src.db.point_id import chunk_point_id
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 1536

# Fields that _build_filter can filter on. Each needs a payload index or Qdrant
# falls back to scanning every point in the collection.
PAYLOAD_KEYWORD_FIELDS = ("cancer_type", "section", "study_design", "chunk_type")
PAYLOAD_INTEGER_FIELDS = ("year", "evidence_level")


# ── Shared data classes ───────────────────────────────────────────────────────

@dataclass
class ChunkDocument:
    """A chunk ready for upsert into Qdrant (vector pre-computed)."""
    chunk_id: str
    text: str
    dense_vector: list[float]
    # ── Metadata fields stored as payload ────────────────────────────────
    pmid: str
    pmcid: str
    title: str
    authors: list[str]
    journal: str
    year: Optional[int]
    cancer_type: list[str]
    section: str
    chunk_type: str
    chunk_index: int
    study_design: str
    sample_size: Optional[int]
    primary_outcome: Optional[str]
    evidence_level: int


@dataclass
class ScoredChunk:
    """A chunk returned by any search method, with a relevance score."""
    chunk_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


# ── QdrantStore ───────────────────────────────────────────────────────────────

class QdrantStore:
    """
    Domain wrapper around QdrantClient.

    Construct with an existing QdrantClient (pass QdrantClient(":memory:") for
    tests or the real client for production).  `ensure_collection` is called
    automatically in __init__.

    `collection_name` is required. It used to default to a module constant
    reading "urological_oncology_v2", which is not the name of any collection
    that exists -- production uses "urological_oncology_papers", from
    QDRANT_COLLECTION. Since ensure_collection creates a missing collection,
    relying on that default would silently produce an empty one and serve an
    empty corpus. Every caller already passes the name explicitly, so requiring
    it costs nothing and removes the failure mode rather than relabelling it.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
    ) -> None:
        self._client = client
        self._collection = collection_name
        self.ensure_collection()

    # ── Collection management ─────────────────────────────────────────────

    def ensure_collection(self) -> None:
        """Create the collection if it is missing, then ensure its payload
        indexes exist.

        The index creation used to sit inside the "collection is missing"
        branch, so it ran only when this class created the collection itself.
        Production never takes that path: the collection is created by
        src/ingestion/embed.ensure_collection, which configures vectors and no
        indexes at all, and QdrantStore then attaches to it. The result was a
        live collection with no payload indexes, confirmed on 2026-09-20, so
        every filtered search scanned all 687,101 points. /treatment-card
        filters on cancer_type for every request.
        """
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
        self._ensure_payload_indexes()

    def _existing_payload_indexes(self) -> set[str]:
        """Field names that already carry a payload index, empty if unknown."""
        try:
            info = self._client.get_collection(collection_name=self._collection)
            return set(getattr(info, "payload_schema", None) or {})
        except Exception:
            logger.debug("Could not read payload schema for %r", self._collection)
            return set()

    def _ensure_payload_indexes(self) -> None:
        """Create any missing payload index, leaving existing ones alone.

        Never raises. This runs on every construction, including at API
        startup, and a Qdrant deployment that refuses index creation (older
        server, restricted credentials, or the in-memory client used by the
        tests, which has no payload indexes at all) must not take the service
        down.

        Requests are made with wait=False so Qdrant builds each index in the
        background. The first production run of this code used the default,
        which blocks until the index is built across every point: all six calls
        timed out after five seconds each, adding 30 seconds to startup and
        logging six warnings claiming the indexes did not exist. Every one of
        them had in fact been created.
        """
        already = self._existing_payload_indexes()
        wanted = [
            *((f, PayloadSchemaType.KEYWORD) for f in PAYLOAD_KEYWORD_FIELDS),
            *((f, PayloadSchemaType.INTEGER) for f in PAYLOAD_INTEGER_FIELDS),
        ]
        requested: list[str] = []
        failed: list[str] = []
        for field_name, schema in wanted:
            if field_name in already:
                continue
            try:
                # wait=False: Qdrant accepts the request and builds the index in
                # the background. Waiting instead makes the client block until
                # the index is built over every point, which on the production
                # collection exceeded the default timeout on all six fields and
                # added 30 seconds to startup for nothing.
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=schema,
                    wait=False,
                )
                requested.append(field_name)
            except Exception as exc:
                failed.append(field_name)
                logger.warning("Could not request payload index on %r: %s", field_name, exc)

        if requested:
            logger.info(
                "Requested payload indexes on %r: %s (built in the background)",
                self._collection, ", ".join(requested),
            )
        if failed:
            # Re-read before claiming anything. A timeout is a client-side wait
            # expiring, not a rejection: Qdrant may well have accepted the
            # request and be building the index anyway. Saying "filtered
            # searches will scan the collection" without checking told operators
            # the opposite of what had actually happened.
            present = self._existing_payload_indexes()
            building = [f for f in failed if f in present]
            missing = [f for f in failed if f not in present]
            if building:
                logger.info(
                    "Index request reported an error but the index exists on: %s",
                    ", ".join(building),
                )
            if missing:
                logger.warning(
                    "No payload index on %s — filtered searches on those fields "
                    "will scan the collection",
                    ", ".join(missing),
                )

    # ── Write ─────────────────────────────────────────────────────────────

    def upsert(self, chunks: list[ChunkDocument], batch_size: int = 100) -> None:
        """Upsert chunks in batches to avoid large single requests."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            points = [_to_point(c) for c in batch]
            self._client.upsert(collection_name=self._collection, points=points)

    def delete_by_pmid(self, pmid: str) -> None:
        """Remove all chunks belonging to a paper (by PMID)."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="pmid", match=MatchValue(value=pmid))]
                )
            ),
        )

    # ── Read ──────────────────────────────────────────────────────────────

    def search_dense(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[ScoredChunk]:
        """ANN search using the dense cosine index."""
        results = self._client.query_points(
            collection_name=self._collection,
            query=query_embedding,
            query_filter=_build_filter(filters),
            limit=top_k,
            with_payload=True,
        )
        return [_to_scored_chunk(p) for p in results.points]

    def scroll_all(self, batch_size: int = 500) -> list[ScoredChunk]:
        """Page through the entire collection (used to build BM25 index)."""
        chunks: list[ScoredChunk] = []
        offset = None
        while True:
            records, next_offset = self._client.scroll(
                collection_name=self._collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                payload = rec.payload or {}
                chunks.append(ScoredChunk(
                    chunk_id=payload.get("chunk_id", str(rec.id)),
                    text=payload.get("text", ""),
                    score=0.0,
                    metadata={k: v for k, v in payload.items()
                               if k not in ("chunk_id", "text")},
                ))
            if next_offset is None:
                break
            offset = next_offset
        return chunks

    def count(self) -> int:
        """Return the current number of points in the collection."""
        return self._client.count(collection_name=self._collection).count

    def collection_stats(self) -> dict:
        """Return point count and collection configuration summary."""
        count = self._client.count(collection_name=self._collection).count
        info = self._client.get_collection(collection_name=self._collection)
        return {
            "collection": self._collection,
            "point_count": count,
            "status": str(info.status),
            "dense_vector_size": EMBEDDING_DIMENSION,
        }


# ── Private helpers ───────────────────────────────────────────────────────────

def _chunk_uuid(chunk_id: str) -> str:
    """Stable UUID derived from the string chunk_id.

    Delegates to src/db/point_id.chunk_point_id, the single definition. This
    used to derive its own id with uuid5, which disagreed with the ingestion
    path's MD5 derivation, so the same chunk written through both paths was
    stored twice rather than updated.
    """
    return chunk_point_id(chunk_id)


def _to_point(c: ChunkDocument) -> PointStruct:
    return PointStruct(
        id=_chunk_uuid(c.chunk_id),
        vector=c.dense_vector,
        payload={
            "chunk_id": c.chunk_id,
            "text": c.text,
            "pmid": c.pmid,
            "pmcid": c.pmcid,
            "title": c.title,
            "authors": c.authors,
            "journal": c.journal,
            "year": c.year,
            "cancer_type": c.cancer_type,
            "section": c.section,
            "chunk_type": c.chunk_type,
            "chunk_index": c.chunk_index,
            "study_design": c.study_design,
            "sample_size": c.sample_size,
            "primary_outcome": c.primary_outcome,
            "evidence_level": c.evidence_level,
        },
    )


def _to_scored_chunk(point) -> ScoredChunk:
    payload = point.payload or {}
    meta = {k: v for k, v in payload.items() if k not in ("chunk_id", "text")}
    return ScoredChunk(
        chunk_id=payload.get("chunk_id", str(point.id)),
        text=payload.get("text", ""),
        score=getattr(point, "score", 0.0),
        metadata=meta,
    )


def _build_filter(filters: dict | None) -> Filter | None:
    if not filters:
        return None
    must: list = []

    for key in ("cancer_type", "section", "study_design", "chunk_type"):
        if key in filters:
            vals = filters[key]
            if isinstance(vals, str):
                vals = [vals]
            must.append(FieldCondition(key=key, match=MatchAny(any=vals)))

    if "year_min" in filters or "year_max" in filters:
        must.append(FieldCondition(
            key="year",
            range=Range(
                gte=filters.get("year_min"),
                lte=filters.get("year_max"),
            ),
        ))

    if "evidence_level_max" in filters:
        must.append(FieldCondition(
            key="evidence_level",
            range=Range(lte=filters["evidence_level_max"]),
        ))

    return Filter(must=must) if must else None
