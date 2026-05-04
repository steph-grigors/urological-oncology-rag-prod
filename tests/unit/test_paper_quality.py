"""
Unit tests for src/ingestion/quality.py and the quality gate in pipeline.py.

All tests are pure (no I/O, no API calls).
Run from project root: pytest tests/unit/test_paper_quality.py -v
"""

from __future__ import annotations

import json
import pytest

from src.ingestion.parse import ParsedPaper, Section
from src.ingestion.quality import (
    MIN_ABSTRACT_WORDS,
    MIN_BODY_WORDS,
    MIN_DISTINCT_SECTIONS,
    PASS_SCORE,
    QualityResult,
    score_paper_quality,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


def _section(name: str, words: int, section_type: str = "text") -> Section:
    return Section(name=name, raw_name=name, content=_words(words), section_type=section_type)


def _paper(
    *,
    abstract: str = "",
    sections: list[Section] | None = None,
) -> ParsedPaper:
    secs: list[Section] = []
    if abstract:
        secs.append(Section(name="abstract", raw_name="Abstract",
                            content=abstract, section_type="text"))
    secs.extend(sections or [])
    return ParsedPaper(
        pmc_id="PMC000001",
        pmid="12345678",
        doi="10.0000/test",
        title="Test Oncology Paper",
        abstract=abstract,
        journal="J Urology",
        year=2023,
        authors=["Smith J"],
        sections=secs,
    )


def _well_formed_paper() -> ParsedPaper:
    """A realistic paper that should pass every check."""
    return _paper(
        abstract=_words(MIN_ABSTRACT_WORDS + 20),
        sections=[
            _section("introduction", 80),
            _section("methods",      MIN_BODY_WORDS // 3),
            _section("results",      MIN_BODY_WORDS // 3),
            _section("discussion",   MIN_BODY_WORDS // 3),
            _section("conclusion",   60),
        ],
    )


# ── score_paper_quality — core scoring ───────────────────────────────────────

class TestScoreRange:
    def test_score_between_0_and_1(self):
        for paper in [
            _well_formed_paper(),
            _paper(),                          # completely empty
            _paper(abstract=_words(10)),       # abstract only
        ]:
            r = score_paper_quality(paper)
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of range"

    def test_perfect_paper_has_maximum_score(self):
        r = score_paper_quality(_well_formed_paper())
        assert r.score == 1.0

    def test_empty_paper_has_minimum_score(self):
        r = score_paper_quality(_paper())
        assert r.score == 0.0


# ── Pass / fail threshold ─────────────────────────────────────────────────────

class TestPassFail:
    def test_well_formed_paper_passes(self):
        assert score_paper_quality(_well_formed_paper()).passed is True

    def test_passed_flag_matches_threshold(self):
        """passed must equal score >= PASS_SCORE for every paper."""
        papers = [
            _well_formed_paper(),
            _paper(),
            _paper(abstract=_words(10)),
            _paper(abstract=_words(MIN_ABSTRACT_WORDS + 5),
                   sections=[_section("results", MIN_BODY_WORDS + 50)]),
        ]
        for paper in papers:
            r = score_paper_quality(paper)
            assert r.passed == (r.score >= PASS_SCORE)

    def test_abstract_only_paper_fails(self):
        """Full-text fetch failed — only abstract available, no body."""
        paper = _paper(abstract=_words(MIN_ABSTRACT_WORDS + 20))
        r = score_paper_quality(paper)
        assert r.passed is False

    def test_body_below_minimum_fails(self):
        # Short body AND no findings — both body_content and results_or_discussion
        # fail, so score is too low and the required check blocks it.
        paper = _paper(
            abstract=_words(MIN_ABSTRACT_WORDS + 20),
            sections=[
                _section("introduction", MIN_BODY_WORDS // 4),
                _section("methods",      MIN_BODY_WORDS // 4),
            ],
        )
        r = score_paper_quality(paper)
        assert not r.passed
        assert any("Body content" in reason for reason in r.reasons)

    def test_no_findings_sections_fails(self):
        """Paper has intro + methods but no results, discussion, or conclusion."""
        paper = _paper(
            abstract=_words(MIN_ABSTRACT_WORDS + 20),
            sections=[
                _section("introduction", MIN_BODY_WORDS // 2),
                _section("methods",      MIN_BODY_WORDS // 2),
            ],
        )
        r = score_paper_quality(paper)
        assert not r.passed
        assert any("results" in reason.lower() for reason in r.reasons)

    def test_single_body_section_penalised(self):
        """Only one distinct section — likely a parsing failure."""
        paper = _paper(
            abstract=_words(MIN_ABSTRACT_WORDS + 10),
            sections=[_section("results", MIN_BODY_WORDS + 50)],
        )
        r = score_paper_quality(paper)
        # section_variety check fails; total score = 0.15+0.10+0.35+0.30 = 0.90
        # Wait — with 1 section, section_variety fails (-0.10). Score = 0.90.
        # Actually this paper should PASS (score 0.90 > 0.50) but with a warning.
        # The section_variety check alone is not enough to fail a paper with
        # good body content and findings. Verify the score is reduced.
        assert r.score < 1.0
        assert any("section" in reason.lower() for reason in r.reasons)


# ── Individual check behaviour ────────────────────────────────────────────────

class TestIndividualChecks:
    def test_missing_abstract_reduces_score(self):
        with_abstract = _well_formed_paper()
        without_abstract = _paper(
            abstract="",
            sections=[
                _section("introduction", 80),
                _section("methods", MIN_BODY_WORDS // 3),
                _section("results", MIN_BODY_WORDS // 3),
                _section("discussion", MIN_BODY_WORDS // 3),
            ],
        )
        assert score_paper_quality(without_abstract).score < score_paper_quality(with_abstract).score

    def test_short_abstract_reduces_score(self):
        long_abstract = _paper(
            abstract=_words(MIN_ABSTRACT_WORDS + 30),
            sections=[
                _section("methods", MIN_BODY_WORDS // 2),
                _section("results", MIN_BODY_WORDS // 2),
            ],
        )
        short_abstract = _paper(
            abstract=_words(MIN_ABSTRACT_WORDS - 10),
            sections=[
                _section("methods", MIN_BODY_WORDS // 2),
                _section("results", MIN_BODY_WORDS // 2),
            ],
        )
        assert score_paper_quality(short_abstract).score < score_paper_quality(long_abstract).score

    def test_results_section_satisfies_findings_check(self):
        paper = _paper(
            abstract=_words(MIN_ABSTRACT_WORDS),
            sections=[
                _section("methods", MIN_BODY_WORDS // 2),
                _section("results", MIN_BODY_WORDS // 2),
            ],
        )
        r = score_paper_quality(paper)
        assert not any("results" in reason.lower() for reason in r.reasons)

    def test_discussion_section_satisfies_findings_check(self):
        paper = _paper(
            abstract=_words(MIN_ABSTRACT_WORDS),
            sections=[
                _section("introduction", MIN_BODY_WORDS // 2),
                _section("discussion",   MIN_BODY_WORDS // 2),
            ],
        )
        r = score_paper_quality(paper)
        assert not any("results" in reason.lower() for reason in r.reasons)

    def test_conclusion_section_satisfies_findings_check(self):
        paper = _paper(
            abstract=_words(MIN_ABSTRACT_WORDS),
            sections=[
                _section("methods",    MIN_BODY_WORDS // 2),
                _section("conclusion", MIN_BODY_WORDS // 2),
            ],
        )
        r = score_paper_quality(paper)
        assert not any("results" in reason.lower() for reason in r.reasons)

    def test_other_sections_excluded_from_variety_count(self):
        """Sections labelled 'other' don't count toward MIN_DISTINCT_SECTIONS."""
        paper = _paper(
            abstract=_words(MIN_ABSTRACT_WORDS),
            sections=[
                _section("other",   MIN_BODY_WORDS // 2),
                _section("results", MIN_BODY_WORDS // 2),
            ],
        )
        r = score_paper_quality(paper)
        # distinct_sections = {"results"} → only 1, so section_variety fails
        assert any("section" in reason.lower() for reason in r.reasons)


# ── Reasons list ──────────────────────────────────────────────────────────────

class TestReasons:
    def test_reasons_empty_when_paper_passes_all_checks(self):
        r = score_paper_quality(_well_formed_paper())
        assert r.reasons == []

    def test_one_reason_per_failing_check(self):
        # abstract missing → abstract_present and abstract_length both fail
        paper = _paper(
            abstract="",
            sections=[
                _section("methods", MIN_BODY_WORDS // 2),
                _section("results", MIN_BODY_WORDS // 2),
            ],
        )
        r = score_paper_quality(paper)
        assert len(r.reasons) == 2

    def test_reasons_mention_actual_values(self):
        """Failure messages must include the observed count, not just the threshold."""
        body_words = 50  # well below MIN_BODY_WORDS
        paper = _paper(
            abstract=_words(MIN_ABSTRACT_WORDS + 10),
            sections=[
                _section("results", body_words),
            ],
        )
        r = score_paper_quality(paper)
        body_reason = next(
            (reason for reason in r.reasons if "Body" in reason), None
        )
        assert body_reason is not None
        assert str(body_words) in body_reason


# ── Pipeline quality gate integration ────────────────────────────────────────

class TestPipelineQualityGate:
    """Verify skip_low_quality wires into run_ingestion correctly."""

    def _setup_monkeypatch(self, monkeypatch, fake_paper: ParsedPaper):
        import src.ingestion.parse as parse_mod

        monkeypatch.setattr(
            "src.ingestion.pipeline.search_pmc",
            lambda *a, **kw: ["PMC001"],
        )
        monkeypatch.setattr(
            "src.ingestion.pipeline.fetch_batch",
            lambda ids, **kw: iter([("PMC001", "<xml/>")]),
        )
        monkeypatch.setattr(parse_mod, "parse_paper", lambda xml: fake_paper)

    def test_low_quality_paper_skipped_by_default(
        self, tmp_path, monkeypatch
    ):
        """A paper that fails the quality gate must not be chunked or embedded."""
        # Abstract-only paper — will fail body_content and results checks
        bad_paper = _paper(abstract=_words(MIN_ABSTRACT_WORDS + 10))
        self._setup_monkeypatch(monkeypatch, bad_paper)

        embed_calls: list = []
        monkeypatch.setattr(
            "src.ingestion.pipeline.embed_chunks",
            lambda *a, **kw: embed_calls.append(True),
        )

        checkpoint = tmp_path / "state.json"
        checkpoint.write_text(json.dumps({"ingested_ids": []}))

        from src.ingestion.pipeline import run_ingestion
        summary = run_ingestion(
            topics=["prostate"],
            skip_low_quality=True,
            checkpoint_path=str(checkpoint),
            rejected_path=str(tmp_path / "rejected.json"),
            dry_run=False,
            openai_client=None,
            qdrant_client=None,
        )

        assert summary.topics["prostate"].chunks_produced == 0
        assert len(embed_calls) == 0

    def test_low_quality_paper_ingested_when_gate_disabled(
        self, tmp_path, monkeypatch
    ):
        """With skip_low_quality=False, even a bad paper reaches chunking."""
        bad_paper = _paper(
            abstract=_words(MIN_ABSTRACT_WORDS + 10),
            sections=[
                _section("results", MIN_BODY_WORDS + 50),
                _section("discussion", 60),
            ],
        )
        self._setup_monkeypatch(monkeypatch, bad_paper)

        checkpoint = tmp_path / "state.json"
        checkpoint.write_text(json.dumps({"ingested_ids": []}))

        from src.ingestion.pipeline import run_ingestion
        summary = run_ingestion(
            topics=["prostate"],
            skip_low_quality=False,
            checkpoint_path=str(checkpoint),
            rejected_path=str(tmp_path / "rejected.json"),
            dry_run=False,
            openai_client=None,
            qdrant_client=None,
        )

        assert summary.topics["prostate"].chunks_produced > 0

    def test_rejected_paper_written_to_file(self, tmp_path, monkeypatch):
        """PMC IDs rejected by the quality gate must appear in the rejected file."""
        bad_paper = _paper(abstract=_words(MIN_ABSTRACT_WORDS + 10))
        self._setup_monkeypatch(monkeypatch, bad_paper)

        checkpoint = tmp_path / "state.json"
        checkpoint.write_text(json.dumps({"ingested_ids": []}))
        rejected_path = tmp_path / "rejected.json"

        from src.ingestion.pipeline import run_ingestion
        run_ingestion(
            topics=["prostate"],
            skip_low_quality=True,
            checkpoint_path=str(checkpoint),
            rejected_path=str(rejected_path),
            dry_run=False,
            openai_client=None,
            qdrant_client=None,
        )

        assert rejected_path.exists()
        records = json.loads(rejected_path.read_text())
        assert any(r["pmc_id"] == "PMC001" for r in records)
        assert all("reasons" in r for r in records)

    def test_high_quality_paper_not_rejected(self, tmp_path, monkeypatch):
        """A passing paper must not appear in the rejected file."""
        good_paper = _well_formed_paper()
        self._setup_monkeypatch(monkeypatch, good_paper)

        checkpoint = tmp_path / "state.json"
        checkpoint.write_text(json.dumps({"ingested_ids": []}))
        rejected_path = tmp_path / "rejected.json"

        from src.ingestion.pipeline import run_ingestion
        run_ingestion(
            topics=["prostate"],
            skip_low_quality=True,
            checkpoint_path=str(checkpoint),
            rejected_path=str(rejected_path),
            dry_run=False,
            openai_client=None,
            qdrant_client=None,
        )

        if rejected_path.exists():
            records = json.loads(rejected_path.read_text())
            assert not any(r["pmc_id"] == "PMC001" for r in records)
