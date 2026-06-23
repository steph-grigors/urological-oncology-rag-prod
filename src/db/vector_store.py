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

import uuid
from dataclasses import dataclass, field
from typing import Optional

from qdrant_client import QdrantClient
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

COLLECTION_NAME = "urological_oncology_v2"
EMBEDDING_DIMENSION = 1536


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
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self._client = client
        self._collection = collection_name
        self.ensure_collection()

    # ── Collection management ─────────────────────────────────────────────

    def ensure_collection(self) -> None:
        """Create collection + payload indexes if they do not already exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            self._create_payload_indexes()

    def _create_payload_indexes(self) -> None:
        keyword_fields = ["cancer_type", "section", "study_design", "chunk_type"]
        integer_fields = ["year", "evidence_level"]
        for f in keyword_fields:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=f,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        for f in integer_fields:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=f,
                field_schema=PayloadSchemaType.INTEGER,
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
    """Stable UUID derived from the string chunk_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


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
