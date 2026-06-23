"""
Unit tests for PubMedWebSearch (cRAG-lite's web-search fallback).

Bio.Entrez is mocked — these tests never hit the live NCBI API.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.retrieval.web_fallback import PubMedWebSearch

_RECORD = {
    "MedlineCitation": {
        "PMID": "12345678",
        "Article": {
            "ArticleTitle": "Enzalutamide in metastatic prostate cancer",
            "Abstract": {"AbstractText": ["Background.", " Results show benefit."]},
            "Journal": {"JournalIssue": {"PubDate": {"Year": "2023"}}},
        },
    }
}


class TestAvailability:
    def test_unavailable_without_email(self):
        assert PubMedWebSearch(email="").is_available() is False

    def test_available_with_email(self):
        assert PubMedWebSearch(email="me@example.com").is_available() is True

    def test_search_without_email_returns_empty_without_calling_entrez(self):
        search = PubMedWebSearch(email="")
        assert search.search("query") == []


class TestSearch:
    def test_no_ids_returns_empty(self):
        search = PubMedWebSearch(email="me@example.com")
        with patch("Bio.Entrez.esearch"), patch(
            "Bio.Entrez.read", return_value={"IdList": []}
        ):
            assert search.search("query") == []

    def test_parses_record_into_ranked_chunk(self):
        search = PubMedWebSearch(email="me@example.com")
        with patch("Bio.Entrez.esearch"), patch("Bio.Entrez.efetch"), patch(
            "Bio.Entrez.read",
            side_effect=[
                {"IdList": ["12345678"]},
                {"PubmedArticle": [_RECORD]},
            ],
        ):
            chunks = search.search("enzalutamide")

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.chunk_id == "pubmed:12345678"
        assert "Background." in chunk.text
        assert chunk.metadata["study_design"] == "web_search"
        assert chunk.metadata["pmid"] == "12345678"
        assert chunk.metadata["year"] == 2023

    def test_record_without_abstract_is_skipped(self):
        record = {
            "MedlineCitation": {
                "PMID": "1",
                "Article": {"ArticleTitle": "No abstract here", "Journal": {}},
            }
        }
        search = PubMedWebSearch(email="me@example.com")
        with patch("Bio.Entrez.esearch"), patch("Bio.Entrez.efetch"), patch(
            "Bio.Entrez.read",
            side_effect=[{"IdList": ["1"]}, {"PubmedArticle": [record]}],
        ):
            assert search.search("query") == []

    def test_entrez_exception_returns_empty_list(self):
        search = PubMedWebSearch(email="me@example.com")
        with patch("Bio.Entrez.esearch", side_effect=RuntimeError("network down")):
            assert search.search("query") == []
