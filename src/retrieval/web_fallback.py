"""
PubMed web-search fallback for cRAG-lite.

When every locally retrieved chunk grades Incorrect (relevance_score below
CONFIDENCE_LOW after reranking), RAGRetriever.retrieve() asks PubMed directly
for the original clinical question via NCBI's E-utilities — the same Entrez
API already used by scripts/data_collection_scaled.py for corpus ingestion.
This restricts the fallback to a single trusted, already-integrated source
rather than the open web, and runs as one deterministic call (no agentic
loop, no per-chunk search).

Graceful degradation: if `email` is empty or the API call fails, `search()`
returns an empty list and RAGRetriever falls through to its existing
REFUSED confidence gate.
"""

from __future__ import annotations

import logging

from config.constants import CONFIDENCE_LOW
from src.retrieval.reranker import RankedChunk

logger = logging.getLogger(__name__)


class PubMedWebSearch:
    """
    Wraps NCBI Entrez esearch + efetch.

    Pass email="" to disable (matches CohereReranker's api_key="" convention
    for graceful degradation).
    """

    def __init__(self, email: str, api_key: str = "") -> None:
        self._email = email
        self._api_key = api_key

    def is_available(self) -> bool:
        return bool(self._email)

    def search(self, query: str, max_results: int = 5) -> list[RankedChunk]:
        if not self.is_available():
            return []

        try:
            from Bio import Entrez

            Entrez.email = self._email
            if self._api_key:
                Entrez.api_key = self._api_key

            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            ids = Entrez.read(handle)["IdList"]
            handle.close()
            if not ids:
                return []

            handle = Entrez.efetch(
                db="pubmed", id=ids, rettype="abstract", retmode="xml"
            )
            records = Entrez.read(handle).get("PubmedArticle", [])
            handle.close()

            chunks = [_record_to_chunk(r) for r in records]
            return [c for c in chunks if c is not None]
        except Exception:
            logger.exception("PubMed web-search fallback failed")
            return []


def _record_to_chunk(record: dict) -> RankedChunk | None:
    try:
        article = record["MedlineCitation"]["Article"]
        pmid = str(record["MedlineCitation"]["PMID"])
        title = str(article.get("ArticleTitle", ""))
        abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
        abstract = " ".join(str(p) for p in abstract_parts)
        if not abstract:
            return None

        year_raw = (
            article.get("Journal", {})
            .get("JournalIssue", {})
            .get("PubDate", {})
            .get("Year")
        )
        year = int(year_raw) if year_raw and str(year_raw).isdigit() else None

        return RankedChunk(
            chunk_id=f"pubmed:{pmid}",
            text=f"{title}\n\n{abstract}",
            score=CONFIDENCE_LOW,
            relevance_score=CONFIDENCE_LOW,
            metadata={
                "title": title,
                "pmid": pmid,
                "year": year,
                "study_design": "web_search",
                "section": "abstract",
            },
        )
    except Exception:
        return None
