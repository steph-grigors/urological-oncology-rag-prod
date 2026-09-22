"""
PubMed web-search fallback for cRAG-lite.

When every locally retrieved chunk grades Incorrect (relevance_score below
CONFIDENCE_LOW after reranking), RAGRetriever.retrieve() asks PubMed directly
for the original clinical question via NCBI's E-utilities — the same Entrez
API already used by scripts/data_collection_scaled.py for corpus ingestion.
This restricts the fallback to a single trusted, already-integrated source
rather than the open web, and runs as one deterministic call (no agentic
loop, no per-chunk search).

Scoring: this returns UNRANKED results. Every abstract used to be assigned
exactly CONFIDENCE_LOW (0.45) -- the threshold value -- so a live PubMed hit
that had never been judged for relevance arrived looking exactly as confident
as the weakest chunk the local corpus is allowed to return. Downstream,
retrieval_confidence became exactly 0.45 whenever this fired, a number nobody
had measured. RAGRetriever now reranks these through the same Cohere
cross-encoder the local corpus goes through.

Availability: `search()` returns an empty list when `email` is empty, and
RAGRetriever falls through to the ungrounded path. Note that the email is a
contact address NCBI asks callers to send, not a credential -- Biopython's
Entrez wrapper simply refuses to issue a request without one. Using it as the
on/off switch means a configuration oversight and a deliberate "off" look
identical; see `enabled` on RAGRetriever for the explicit switch.
"""

from __future__ import annotations

import logging

from src.db.vector_store import ScoredChunk

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

    def search(self, query: str, max_results: int = 5) -> list[ScoredChunk]:
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


def _record_to_chunk(record: dict) -> ScoredChunk | None:
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

        return ScoredChunk(
            chunk_id=f"pubmed:{pmid}",
            text=f"{title}\n\n{abstract}",
            # No score here. These are unranked search hits; the caller passes
            # them through the same reranker the local corpus goes through, so
            # they arrive with a measured relevance rather than an assumed one.
            score=0.0,
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
