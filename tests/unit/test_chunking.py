"""
Unit tests for src/ingestion/chunk.py, src/ingestion/parse.normalize_section,
src/ingestion/pipeline.run_ingestion, src/ingestion/embed (cost estimate),
and scripts/migrate_chromadb_to_qdrant (SQLite reader).

Run from project root: pytest tests/unit/test_chunking.py -v
"""

import json
import pytest

from src.ingestion.chunk import (
    Chunk,
    ChunkMetadata,
    EVIDENCE_LEVELS,
    SHORT_SECTION_THRESHOLD,
    TABLE_TRUNCATION_NOTE,
    TABLE_WORD_CAP,
    chunk_paper,
    chunk_section,
)
from src.ingestion.parse import ParsedPaper, Section, normalize_section


# ── Helpers ───────────────────────────────────────────────────────────────────

def _words(n: int) -> str:
    """Build a deterministic n-word string: 'word0 word1 ... word{n-1}'."""
    return " ".join(f"word{i}" for i in range(n))


def _section(
    content: str,
    *,
    section_type: str = "text",
    name: str = "results",
) -> Section:
    return Section(name=name, raw_name=name, content=content, section_type=section_type)


def _paper(sections: list[Section]) -> ParsedPaper:
    return ParsedPaper(
        pmc_id="PMC000001",
        pmid="11111111",
        doi="10.0000/test",
        title="Prostate Cancer Treatment Study",
        abstract="",
        journal="J Urology",
        year=2023,
        authors=["Smith J", "Doe A"],
        sections=sections,
    )


# Base metadata dict — matches every field in ChunkMetadata except
# section / chunk_type / chunk_index, which are filled by chunk_section.
_BASE = dict(
    pmid="11111111",
    pmcid="PMC000001",
    title="Prostate Cancer Treatment Study",
    authors=["Smith J", "Doe A"],
    journal="J Urology",
    year=2023,
    cancer_type=["prostate"],
    study_design="rct",
    sample_size=120,
    primary_outcome="Overall survival at 3 years",
    intervention="enzalutamide 160 mg/day",
    comparator="placebo",
    evidence_level=EVIDENCE_LEVELS["rct"],
)


# ── Section detection (normalize_section) ─────────────────────────────────────

class TestNormalizeSection:
    """parse.normalize_section maps heading text to canonical labels."""

    def test_abstract(self):
        assert normalize_section("Abstract") == "abstract"
        assert normalize_section("ABSTRACT") == "abstract"
        assert normalize_section("Summary") == "abstract"

    def test_introduction(self):
        assert normalize_section("Introduction") == "introduction"
        assert normalize_section("Background") == "introduction"
        assert normalize_section("BACKGROUND AND RATIONALE") == "introduction"

    def test_methods(self):
        assert normalize_section("Methods") == "methods"
        assert normalize_section("Materials and Methods") == "methods"
        assert normalize_section("Patients and Methods") == "methods"
        assert normalize_section("Methodology") == "methods"
        assert normalize_section("Study Design") == "methods"

    def test_results(self):
        assert normalize_section("Results") == "results"
        assert normalize_section("Findings") == "results"
        assert normalize_section("Clinical Outcomes") == "results"

    def test_discussion(self):
        assert normalize_section("Discussion") == "discussion"
        assert normalize_section("DISCUSSION") == "discussion"

    def test_conclusion(self):
        assert normalize_section("Conclusion") == "conclusion"
        assert normalize_section("Conclusions") == "conclusion"
        assert normalize_section("Concluding Remarks") == "conclusion"

    def test_references_returns_skip(self):
        # References sections are flagged for omission
        assert normalize_section("References") == "_skip"
        assert normalize_section("Bibliography") == "_skip"

    def test_acknowledgements_returns_skip(self):
        assert normalize_section("Acknowledgements") == "_skip"
        assert normalize_section("Funding") == "_skip"
        assert normalize_section("Conflict of Interest") == "_skip"

    def test_unknown_heading_returns_other(self):
        assert normalize_section("Appendix A") == "other"
        assert normalize_section("Supplemental Data") == "_skip"

    def test_empty_string_returns_other(self):
        assert normalize_section("") == "other"

    def test_case_insensitive(self):
        assert normalize_section("METHODS") == "methods"
        assert normalize_section("Results") == normalize_section("RESULTS")


# ── Short-section rule ────────────────────────────────────────────────────────

class TestShortSection:
    """Sections at or below SHORT_SECTION_THRESHOLD words → single chunk."""

    def test_one_word_below_min_chunk_words_is_dropped(self):
        # 1 word < MIN_CHUNK_WORDS (30) — filtered by the minimum length guard
        chunks = chunk_section(_section("sunitinib"), _BASE, start_index=0)
        assert len(chunks) == 0

    def test_below_threshold_is_single_chunk(self):
        chunks = chunk_section(
            _section(_words(SHORT_SECTION_THRESHOLD - 1)), _BASE, start_index=0
        )
        assert len(chunks) == 1

    def test_at_threshold_is_single_chunk(self):
        chunks = chunk_section(
            _section(_words(SHORT_SECTION_THRESHOLD)), _BASE, start_index=0
        )
        assert len(chunks) == 1

    def test_short_chunk_preserves_full_text(self):
        content = _words(40)
        chunks = chunk_section(_section(content), _BASE, start_index=0)
        assert chunks[0].text == content

    def test_empty_content_produces_no_chunks(self):
        chunks = chunk_section(_section(""), _BASE, start_index=0)
        assert chunks == []

    def test_whitespace_only_produces_no_chunks(self):
        chunks = chunk_section(_section("   \n  \t  "), _BASE, start_index=0)
        assert chunks == []


# ── Table handling ────────────────────────────────────────────────────────────

class TestTableChunks:
    """Tables are always a single chunk regardless of length."""

    def test_small_table_is_single_chunk(self):
        chunks = chunk_section(
            _section(_words(50), section_type="table"), _BASE, start_index=0
        )
        assert len(chunks) == 1

    def test_large_table_is_still_single_chunk(self):
        # 350 words > default chunk_size=200 but must not be split
        chunks = chunk_section(
            _section(_words(350), section_type="table"), _BASE,
            start_index=0, chunk_size=200,
        )
        assert len(chunks) == 1

    def test_table_chunk_type_is_table(self):
        chunks = chunk_section(
            _section(_words(50), section_type="table"), _BASE, start_index=0
        )
        assert chunks[0].metadata.chunk_type == "table"

    def test_table_over_cap_is_truncated(self):
        chunks = chunk_section(
            _section(_words(TABLE_WORD_CAP + 100), section_type="table"),
            _BASE, start_index=0,
        )
        assert TABLE_TRUNCATION_NOTE in chunks[0].text

    def test_truncated_table_word_count_bounded(self):
        chunks = chunk_section(
            _section(_words(TABLE_WORD_CAP + 100), section_type="table"),
            _BASE, start_index=0,
        )
        # TABLE_WORD_CAP words + note (a few words) — must not be much longer
        word_count = len(chunks[0].text.split())
        assert word_count <= TABLE_WORD_CAP + 10

    def test_table_under_cap_not_truncated(self):
        chunks = chunk_section(
            _section(_words(TABLE_WORD_CAP - 1), section_type="table"),
            _BASE, start_index=0,
        )
        assert TABLE_TRUNCATION_NOTE not in chunks[0].text

    def test_figure_caption_is_single_chunk(self):
        chunks = chunk_section(
            _section("Figure 1: Kaplan-Meier overall survival curves.",
                     section_type="figure_caption"),
            _BASE, start_index=0,
        )
        assert len(chunks) == 1
        assert chunks[0].metadata.chunk_type == "figure_caption"


# ── Chunk IDs ─────────────────────────────────────────────────────────────────

class TestChunkIds:
    """IDs must be stable, unique, and follow the pmcid_section_index pattern."""

    def test_id_format(self):
        sec = _section(_words(40), name="methods")
        chunks = chunk_section(sec, _BASE, start_index=7)
        assert chunks[0].id == "PMC000001_methods_7"

    def test_ids_are_stable_across_calls(self):
        content = _words(250)
        sec = _section(content, name="results")
        run1 = chunk_section(sec, _BASE, start_index=0)
        run2 = chunk_section(sec, _BASE, start_index=0)
        assert [c.id for c in run1] == [c.id for c in run2]

    def test_ids_unique_within_paper(self):
        paper = _paper([
            _section(_words(250), name="methods"),
            _section(_words(250), name="results"),
            _section(_words(250), name="discussion"),
        ])
        chunks = chunk_paper(paper, cancer_type=["prostate"])
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_start_index_offset_in_id(self):
        """chunk_section with start_index=10 → first ID contains index 10."""
        sec = _section(_words(40), name="discussion")
        chunks = chunk_section(sec, _BASE, start_index=10)
        assert chunks[0].id == "PMC000001_discussion_10"

    def test_global_chunk_index_monotonically_increases(self):
        paper = _paper([
            _section(_words(250), name="results"),
            _section(_words(250), name="discussion"),
        ])
        chunks = chunk_paper(paper, cancer_type=["prostate"])
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == sorted(indices)
        assert indices == list(range(len(indices)))


# ── Overlap correctness ───────────────────────────────────────────────────────

class TestOverlap:
    """The last `overlap` words of chunk N must equal the first `overlap` words
    of chunk N+1, and no words from the original section are lost."""

    # Use chunk_size=40, overlap=8 with 100 words so 100 > SHORT_SECTION_THRESHOLD=80
    _CHUNK_SIZE = 40
    _OVERLAP = 8
    _N_WORDS = 100

    def _make_chunks(self) -> list[Chunk]:
        content = _words(self._N_WORDS)
        sec = _section(content)
        return chunk_section(sec, _BASE, start_index=0,
                             chunk_size=self._CHUNK_SIZE, overlap=self._OVERLAP)

    def test_produces_multiple_chunks(self):
        assert len(self._make_chunks()) > 1

    def test_overlap_words_match_at_every_boundary(self):
        chunks = self._make_chunks()
        for i in range(len(chunks) - 1):
            tail = chunks[i].text.split()[-self._OVERLAP:]
            head = chunks[i + 1].text.split()[:self._OVERLAP]
            assert tail == head, (
                f"Boundary {i}→{i+1}: tail={tail!r}  head={head!r}"
            )

    def test_no_word_is_lost(self):
        content = _words(self._N_WORDS)
        sec = _section(content)
        chunks = chunk_section(sec, _BASE, start_index=0,
                               chunk_size=self._CHUNK_SIZE, overlap=self._OVERLAP)
        recovered = set()
        for c in chunks:
            recovered.update(c.text.split())
        assert recovered == set(content.split())

    def test_no_chunk_exceeds_chunk_size(self):
        chunks = self._make_chunks()
        for c in chunks:
            assert len(c.text.split()) <= self._CHUNK_SIZE

    def test_custom_overlap_respected(self):
        """Overlap=5 produces different chunk boundaries than overlap=15."""
        content = _words(100)
        sec = _section(content)
        chunks5 = chunk_section(sec, _BASE, start_index=0,
                                chunk_size=40, overlap=5)
        chunks15 = chunk_section(sec, _BASE, start_index=0,
                                 chunk_size=40, overlap=15)
        # More overlap → more chunks (smaller step → more iterations)
        assert len(chunks15) >= len(chunks5)


# ── Metadata completeness ─────────────────────────────────────────────────────

class TestMetadataFields:
    """Every chunk must carry all fields defined in the metadata schema."""

    _REQUIRED_FIELDS = frozenset({
        "pmid", "pmcid", "title", "authors", "journal", "year",
        "cancer_type", "section", "chunk_type", "chunk_index",
        "study_design", "sample_size", "primary_outcome", "evidence_level",
        "intervention", "comparator",
    })

    def _first_chunk(self) -> Chunk:
        return chunk_section(_section(_words(50)), _BASE, start_index=0)[0]

    def test_all_required_fields_present(self):
        meta = self._first_chunk().metadata.__dict__
        missing = self._REQUIRED_FIELDS - meta.keys()
        assert not missing, f"Missing metadata fields: {missing}"

    def test_evidence_level_derived_from_study_design(self):
        for design, expected in EVIDENCE_LEVELS.items():
            base = {**_BASE, "study_design": design,
                    "evidence_level": EVIDENCE_LEVELS[design]}
            chunks = chunk_section(_section(_words(40)), base, start_index=0)
            assert chunks[0].metadata.evidence_level == expected

    def test_section_label_propagated(self):
        sec = _section(_words(40), name="discussion")
        chunks = chunk_section(sec, _BASE, start_index=0)
        assert chunks[0].metadata.section == "discussion"

    def test_cancer_type_propagated(self):
        chunks = chunk_section(_section(_words(40)), _BASE, start_index=0)
        assert chunks[0].metadata.cancer_type == ["prostate"]

    def test_chunk_type_text_for_text_section(self):
        chunks = chunk_section(_section(_words(40)), _BASE, start_index=0)
        assert chunks[0].metadata.chunk_type == "text"


# ── chunk_paper integration ───────────────────────────────────────────────────

class TestChunkPaper:
    """chunk_paper wires sections together with correct global indexing."""

    def test_empty_paper_returns_no_chunks(self):
        assert chunk_paper(_paper([]), cancer_type=["prostate"]) == []

    def test_single_short_section_one_chunk(self):
        paper = _paper([_section(_words(40), name="results")])
        chunks = chunk_paper(paper, cancer_type=["kidney"])
        assert len(chunks) == 1

    def test_cancer_type_passed_through(self):
        paper = _paper([_section(_words(40), name="results")])
        chunks = chunk_paper(paper, cancer_type=["bladder", "kidney"])
        assert chunks[0].metadata.cancer_type == ["bladder", "kidney"]

    def test_study_design_and_evidence_level_passed_through(self):
        paper = _paper([_section(_words(40), name="results")])
        chunks = chunk_paper(paper, cancer_type=["prostate"],
                             study_design="meta_analysis")
        assert chunks[0].metadata.study_design == "meta_analysis"
        assert chunks[0].metadata.evidence_level == EVIDENCE_LEVELS["meta_analysis"]

    def test_unknown_study_design_defaults_level_6(self):
        paper = _paper([_section(_words(40))])
        chunks = chunk_paper(paper, cancer_type=["prostate"])
        assert chunks[0].metadata.evidence_level == 6

    def test_multi_section_paper_chunk_count(self):
        # Each section has exactly SHORT_SECTION_THRESHOLD words → 1 chunk each
        paper = _paper([
            _section(_words(SHORT_SECTION_THRESHOLD), name="methods"),
            _section(_words(SHORT_SECTION_THRESHOLD), name="results"),
            _section(_words(SHORT_SECTION_THRESHOLD), name="discussion"),
        ])
        chunks = chunk_paper(paper, cancer_type=["testicular"])
        assert len(chunks) == 3

    def test_sections_containing_only_whitespace_are_skipped(self):
        paper = _paper([
            _section("   "),
            _section(_words(40), name="results"),
            _section(""),
        ])
        chunks = chunk_paper(paper, cancer_type=["prostate"])
        assert len(chunks) == 1
        assert chunks[0].metadata.section == "results"


# ── Incremental-mode deduplication ────────────────────────────────────────────

class TestIncrementalMode:
    """run_ingestion must skip PMC IDs already recorded in the checkpoint."""

    def test_already_ingested_pmc_id_is_skipped(self, tmp_path, monkeypatch):
        """A PMC ID in the checkpoint is counted as skipped, not fetched."""
        checkpoint = tmp_path / "state.json"
        checkpoint.write_text(json.dumps({"ingested_ids": ["PMC999"]}))

        monkeypatch.setattr(
            "src.ingestion.pipeline.search_pmc",
            lambda *a, **kw: ["PMC999"],  # returns only the already-ingested ID
        )
        monkeypatch.setattr(
            "src.ingestion.pipeline.fetch_batch",
            lambda ids, **kw: iter([]),
        )

        from src.ingestion.pipeline import run_ingestion
        summary = run_ingestion(
            topics=["prostate"],
            checkpoint_path=str(checkpoint),
            dry_run=False,
            openai_client=None,
            qdrant_client=None,
        )

        assert summary.topics["prostate"].papers_skipped == 1
        assert summary.topics["prostate"].papers_fetched == 0

    def test_new_pmc_id_not_in_checkpoint_is_processed(self, tmp_path, monkeypatch):
        """A PMC ID NOT in the checkpoint is fetched (attempt made)."""
        checkpoint = tmp_path / "state.json"
        checkpoint.write_text(json.dumps({"ingested_ids": []}))

        fetch_calls: list[str] = []

        def mock_fetch_batch(ids, **kw):
            fetch_calls.extend(ids)
            return iter([])  # yield nothing (XML unavailable in test)

        monkeypatch.setattr(
            "src.ingestion.pipeline.search_pmc",
            lambda *a, **kw: ["PMC001"],
        )
        monkeypatch.setattr("src.ingestion.pipeline.fetch_batch", mock_fetch_batch)

        from src.ingestion.pipeline import run_ingestion
        run_ingestion(
            topics=["prostate"],
            checkpoint_path=str(checkpoint),
            dry_run=False,
            openai_client=None,
            qdrant_client=None,
        )

        assert "PMC001" in fetch_calls

    def test_checkpoint_updated_after_successful_paper(self, tmp_path, monkeypatch):
        """Checkpoint is written only after a successful embed.
        With no embedding clients the paper must NOT be checkpointed so
        a subsequent run with clients will re-process and embed it.
        """
        import src.ingestion.parse as parse_mod
        from src.ingestion.parse import ParsedPaper, Section

        checkpoint = tmp_path / "state.json"
        checkpoint.write_text(json.dumps({"ingested_ids": []}))

        fake_paper = ParsedPaper(
            pmc_id="PMC42",
            pmid="99",
            doi="",
            title="Test Paper",
            abstract="",
            journal="J Test",
            year=2023,
            authors=["A B"],
            sections=[Section("results", "results", _words(40), "text")],
        )

        monkeypatch.setattr(
            "src.ingestion.pipeline.search_pmc",
            lambda *a, **kw: ["PMC42"],
        )
        monkeypatch.setattr(
            "src.ingestion.pipeline.fetch_batch",
            lambda ids, **kw: iter([("PMC42", "<xml/>")]),
        )
        monkeypatch.setattr(parse_mod, "parse_paper", lambda xml: fake_paper)

        from src.ingestion.pipeline import run_ingestion
        run_ingestion(
            topics=["prostate"],
            checkpoint_path=str(checkpoint),
            skip_low_quality=False,
            dry_run=False,
            openai_client=None,   # no clients → embed skipped → no checkpoint
            qdrant_client=None,
        )

        # Paper must NOT be checkpointed — it was never embedded into Qdrant
        saved = json.loads(checkpoint.read_text())
        assert "PMC42" not in saved["ingested_ids"]


# ── Dry-run guard ─────────────────────────────────────────────────────────────

class TestDryRun:
    """With dry_run=True, embed_chunks must never be called."""

    def test_dry_run_skips_embedding(self, monkeypatch):
        embed_called: list[bool] = []

        monkeypatch.setattr(
            "src.ingestion.pipeline.search_pmc",
            lambda *a, **kw: ["PMC001", "PMC002"],
        )
        monkeypatch.setattr(
            "src.ingestion.pipeline.embed_chunks",
            lambda *a, **kw: embed_called.append(True),
        )

        from src.ingestion.pipeline import run_ingestion
        summary = run_ingestion(
            topics=["prostate"],
            dry_run=True,
            openai_client=object(),  # truthy — would normally trigger embedding
            qdrant_client=object(),
        )

        assert len(embed_called) == 0
        assert summary.topics["prostate"].papers_fetched == 2

    def test_dry_run_reports_would_be_fetched_count(self, monkeypatch):
        """Dry-run total_papers counts the new (not-yet-ingested) IDs found."""
        monkeypatch.setattr(
            "src.ingestion.pipeline.search_pmc",
            lambda *a, **kw: ["PMC1", "PMC2", "PMC3"],
        )

        from src.ingestion.pipeline import run_ingestion
        summary = run_ingestion(topics=["bladder"], dry_run=True)
        assert summary.total_papers == 3


# ── Cost estimate ─────────────────────────────────────────────────────────────

class TestCostEstimate:
    """EmbedSummary.estimated_cost_usd must be proportional to word count."""

    def test_cost_grows_with_chunk_count(self):
        """More chunks → higher estimated cost."""
        from unittest.mock import MagicMock
        from src.ingestion.embed import embed_chunks, EmbedSummary

        mock_openai = MagicMock()
        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value.collections = []

        # 2-word chunk → minimal cost
        class _FakeChunk:
            id = "pmcA_results_0"
            text = "word1 word2"
            class metadata:
                title = "T"; section = "results"; chunk_type = "text"
                chunk_index = 0; pmcid = "A"; pmid = "1"
                authors = []; journal = "J"; year = 2022
                cancer_type = ["prostate"]; study_design = "rct"
                sample_size = None; primary_outcome = None; evidence_level = 2

        mock_openai.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 1536)
        ]

        small = embed_chunks([_FakeChunk()], mock_openai, mock_qdrant, "test_col", batch_size=100)
        assert small.estimated_cost_usd > 0

        # 200-word chunk → higher cost
        class _BigChunk:
            id = "pmcB_results_0"
            text = " ".join(f"word{i}" for i in range(200))
            class metadata:
                title = "T"; section = "results"; chunk_type = "text"
                chunk_index = 0; pmcid = "B"; pmid = "2"
                authors = []; journal = "J"; year = 2022
                cancer_type = ["bladder"]; study_design = "rct"
                sample_size = None; primary_outcome = None; evidence_level = 2

        mock_openai.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 1536)
        ]

        big = embed_chunks([_BigChunk()], mock_openai, mock_qdrant, "test_col", batch_size=100)
        assert big.estimated_cost_usd > small.estimated_cost_usd

    def test_cost_formula(self):
        """Cost for N words = N * 1.3 / 1000 * 0.00002."""
        from src.ingestion.embed import _COST_PER_1K_TOKENS, _TOKENS_PER_WORD
        n_words = 1000
        expected = n_words * _TOKENS_PER_WORD / 1000 * _COST_PER_1K_TOKENS
        assert abs(expected - (1000 * 1.3 / 1000 * 0.00002)) < 1e-10


# ── Migration script (SQLite reader) ─────────────────────────────────────────

class TestMigrationSQLiteReader:
    """migrate_chromadb_to_qdrant.read_chromadb_records reads the real SQLite."""

    SQLITE_PATH = "chroma_db_scaled/chroma.sqlite3"

    def test_record_count_matches_embeddings_table(self):
        import sqlite3 as _sqlite3
        from pathlib import Path

        if not Path(self.SQLITE_PATH).exists():
            pytest.skip("ChromaDB SQLite not present")

        from scripts.migrate_chromadb_to_qdrant import read_chromadb_records
        records = read_chromadb_records(self.SQLITE_PATH)

        conn = _sqlite3.connect(self.SQLITE_PATH)
        expected = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        conn.close()

        assert len(records) == expected

    def test_records_have_required_fields(self):
        from pathlib import Path
        if not Path(self.SQLITE_PATH).exists():
            pytest.skip("ChromaDB SQLite not present")

        from scripts.migrate_chromadb_to_qdrant import read_chromadb_records
        records = read_chromadb_records(self.SQLITE_PATH)

        required = {"embedding_id", "chroma:document", "pmc_id"}
        for rec in records[:20]:
            missing = required - rec.keys()
            assert not missing, f"Record missing keys: {missing}"

    def test_migration_dry_run_no_writes(self):
        """dry_run=True must not call openai or qdrant."""
        from pathlib import Path
        if not Path(self.SQLITE_PATH).exists():
            pytest.skip("ChromaDB SQLite not present")

        from unittest.mock import MagicMock
        from scripts.migrate_chromadb_to_qdrant import migrate

        mock_openai = MagicMock()
        mock_qdrant = MagicMock()

        report = migrate(
            sqlite_path=self.SQLITE_PATH,
            openai_client=mock_openai,
            qdrant_client=mock_qdrant,
            collection="test",
            dry_run=True,
        )

        mock_openai.embeddings.create.assert_not_called()
        mock_qdrant.upsert.assert_not_called()
        assert report.source_count > 0
        assert report.migrated == 0
