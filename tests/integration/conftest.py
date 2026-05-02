"""
Integration test fixtures for the retrieval pipeline.

Uses an in-memory QdrantClient so no external services are required.
Embeddings are deterministic numpy vectors — semantically structured so
that same-cancer-type queries and documents have higher cosine similarity.

The fixture corpus is built by:
  1. Parsing all 10 JATS XML files in tests/fixtures/sample_papers/
  2. Chunking each paper with the production chunker
  3. Assigning deterministic embeddings based on cancer_type + section
  4. Upserting into in-memory Qdrant
  5. Building the BM25 index from the same chunks
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.db.vector_store import (
    ChunkDocument,
    QdrantStore,
    ScoredChunk,
    text_to_sparse_vector,
)
from src.ingestion.chunk import chunk_paper
from src.ingestion.parse import parse_paper
from src.retrieval.bm25_search import BM25Search

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_papers"
EMBEDDING_DIM = 1536
TEST_COLLECTION = "test_urological_oncology"

# ── Study design assignments for each fixture paper (by PMC ID) ───────────────
_PAPER_META: dict[str, dict] = {
    "1001": {"cancer_type": ["prostate"], "study_design": "rct"},
    "1002": {"cancer_type": ["prostate"], "study_design": "rct"},
    "1003": {"cancer_type": ["prostate"], "study_design": "meta_analysis"},
    "1004": {"cancer_type": ["prostate"], "study_design": "case_report"},
    "1005": {"cancer_type": ["prostate"], "study_design": "cohort"},
    "1006": {"cancer_type": ["bladder"],  "study_design": "rct"},
    "1007": {"cancer_type": ["bladder"],  "study_design": "cohort"},
    "1008": {"cancer_type": ["kidney"],   "study_design": "review"},
    "1009": {"cancer_type": ["kidney"],   "study_design": "cohort"},
    "1010": {"cancer_type": ["testicular"], "study_design": "rct"},
}

# Canonical unit vectors per cancer type — used to build structured embeddings
_CANCER_BASES: dict[str, np.ndarray] = {}


def _cancer_basis(cancer_type: str) -> np.ndarray:
    """Return a fixed unit vector for a cancer type (deterministic across runs)."""
    if cancer_type not in _CANCER_BASES:
        rng = np.random.default_rng(abs(hash(cancer_type)) % (2**31))
        v = rng.standard_normal(EMBEDDING_DIM)
        _CANCER_BASES[cancer_type] = v / np.linalg.norm(v)
    return _CANCER_BASES[cancer_type]


def make_embedding(cancer_type: str, chunk_id: str) -> list[float]:
    """
    Deterministic 1536-dim embedding.

    90% weight on the cancer-type basis vector + 10% random noise seeded by
    chunk_id.  Ensures same-cancer-type documents cluster together while
    remaining individually distinguishable.
    """
    basis = _cancer_basis(cancer_type)
    rng = np.random.default_rng(abs(hash(chunk_id)) % (2**31))
    noise = rng.standard_normal(EMBEDDING_DIM)
    noise /= np.linalg.norm(noise)
    v = 0.9 * basis + 0.1 * noise
    return (v / np.linalg.norm(v)).tolist()


def query_embedding_for(cancer_type: str) -> list[float]:
    """Return the pure cancer-type basis vector as a query embedding."""
    return _cancer_basis(cancer_type).tolist()


# ── Session-scoped fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qdrant_store() -> QdrantStore:
    client = QdrantClient(":memory:")
    return QdrantStore(client, collection_name=TEST_COLLECTION)


@pytest.fixture(scope="session")
def indexed_chunks(qdrant_store: QdrantStore) -> list[ScoredChunk]:
    """
    Parse, chunk, embed, and upsert all 10 fixture papers.
    Returns the ScoredChunk list used to build the BM25 index.
    """
    all_docs: list[ChunkDocument] = []
    all_scored: list[ScoredChunk] = []

    for xml_path in sorted(FIXTURES_DIR.glob("PMC*.xml")):
        xml_text = xml_path.read_text(encoding="utf-8")
        paper = parse_paper(xml_text)
        if paper is None:
            continue

        pmc_id = paper.pmc_id
        meta = _PAPER_META.get(pmc_id, {"cancer_type": ["prostate"], "study_design": "unknown"})
        cancer_type = meta["cancer_type"]
        study_design = meta["study_design"]

        chunks = chunk_paper(
            paper,
            cancer_type=cancer_type,
            study_design=study_design,
        )

        for chunk in chunks:
            dense_vec = make_embedding(cancer_type[0], chunk.id)
            sparse_idx, sparse_val = text_to_sparse_vector(chunk.text)

            all_docs.append(ChunkDocument(
                chunk_id=chunk.id,
                text=chunk.text,
                dense_vector=dense_vec,
                sparse_indices=sparse_idx,
                sparse_values=sparse_val,
                pmid=chunk.metadata.pmid,
                pmcid=chunk.metadata.pmcid,
                title=chunk.metadata.title,
                authors=chunk.metadata.authors,
                journal=chunk.metadata.journal,
                year=chunk.metadata.year,
                cancer_type=chunk.metadata.cancer_type,
                section=chunk.metadata.section,
                chunk_type=chunk.metadata.chunk_type,
                chunk_index=chunk.metadata.chunk_index,
                study_design=chunk.metadata.study_design,
                sample_size=chunk.metadata.sample_size,
                primary_outcome=chunk.metadata.primary_outcome,
                evidence_level=chunk.metadata.evidence_level,
            ))
            all_scored.append(ScoredChunk(
                chunk_id=chunk.id,
                text=chunk.text,
                score=0.0,
                metadata={
                    "cancer_type": chunk.metadata.cancer_type,
                    "section": chunk.metadata.section,
                    "study_design": chunk.metadata.study_design,
                    "chunk_type": chunk.metadata.chunk_type,
                    "evidence_level": chunk.metadata.evidence_level,
                    "year": chunk.metadata.year,
                    "pmcid": chunk.metadata.pmcid,
                },
            ))

    qdrant_store.upsert(all_docs)
    return all_scored


@pytest.fixture(scope="session")
def bm25_index(indexed_chunks: list[ScoredChunk]) -> BM25Search:
    return BM25Search(indexed_chunks)
