"""
Tests that the README's documented commands actually exist.

It documented three ingestion steps -- src.ingestion.fetch, src.ingestion.load
and src.ingestion.run_pipeline -- of which two name modules that have never
existed, with flags (--topics, --max-per-topic) the real parser does not
accept. Anyone following it would have hit ModuleNotFoundError on step three.

Prose drifts and that is expected; these tests cover only the parts that can
be checked mechanically.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"


def _documented_module_invocations() -> list[str]:
    """`python -m some.module` occurrences in fenced blocks."""
    return sorted(set(re.findall(r"python -m ([a-zA-Z0-9_.]+)", README.read_text())))


def _documented_scripts() -> list[str]:
    return sorted(set(re.findall(r"python (scripts/[A-Za-z0-9_./-]+\.py)", README.read_text())))


class TestDocumentedCommandsExist:
    def test_every_module_invocation_resolves_to_a_file(self):
        missing = []
        for dotted in _documented_module_invocations():
            path = REPO_ROOT / (dotted.replace(".", "/") + ".py")
            package = REPO_ROOT / dotted.replace(".", "/") / "__init__.py"
            if not path.is_file() and not package.is_file():
                missing.append(dotted)
        assert not missing, f"README documents modules that do not exist: {missing}"

    def test_every_script_invocation_resolves_to_a_file(self):
        missing = [s for s in _documented_scripts() if not (REPO_ROOT / s).is_file()]
        assert not missing, f"README documents scripts that do not exist: {missing}"

    def test_the_ingestion_pipeline_is_documented(self):
        assert "src.ingestion.pipeline" in _documented_module_invocations()

    def test_the_nonexistent_modules_are_not_invoked(self):
        """Checks invocations, not mentions. The README deliberately names
        these two in a sentence explaining that they do not exist, so that
        anyone who saw the old instructions elsewhere knows why they failed.
        Naming them is fine; telling someone to run them is not."""
        invocations = _documented_module_invocations()
        for gone in ("src.ingestion.load", "src.ingestion.run_pipeline"):
            assert gone not in invocations, f"README tells the reader to run {gone}"


class TestDocumentedFlagsAreAccepted:
    """Flags in the README's ingestion examples are parsed by the real parser."""

    def _pipeline_flags_in_readme(self) -> set[str]:
        text = README.read_text()
        found = set()
        for line in text.splitlines():
            if "python -m src.ingestion.pipeline" not in line:
                continue
            found.update(re.findall(r"(--[a-z-]+)", line))
        return found

    def test_readme_shows_at_least_one_example(self):
        assert self._pipeline_flags_in_readme(), "no pipeline example with flags found"

    @pytest.mark.parametrize("flag", ["--since-date", "--since-days", "--dry-run", "--limit"])
    def test_documented_flag_is_accepted_by_the_parser(self, flag):
        from src.ingestion.pipeline import _build_parser

        options = set()
        for action in _build_parser()._actions:
            options.update(action.option_strings)
        assert flag in options, f"README shows {flag} but the parser rejects it"

    def test_no_readme_flag_is_unknown_to_the_parser(self):
        from src.ingestion.pipeline import _build_parser

        options = set()
        for action in _build_parser()._actions:
            options.update(action.option_strings)
        unknown = self._pipeline_flags_in_readme() - options
        assert not unknown, f"README documents flags the parser rejects: {sorted(unknown)}"


class TestCorpusFiguresAreNotTheOldInflatedOnes:
    """The old figures came from summing across ingestion runs, which
    double-counted re-processed papers."""

    @pytest.mark.parametrize("stale", ["815 full-text", "31,361", "795,306"])
    def test_stale_figure_is_gone(self, stale):
        assert stale not in README.read_text()

    def test_all_six_cancer_types_are_named(self):
        text = README.read_text().lower()
        for topic in ("prostate", "bladder", "kidney", "testicular", "penile", "adrenal"):
            assert topic in text, f"{topic} is supported but unmentioned"

    def test_both_endpoints_are_mentioned(self):
        text = README.read_text()
        assert "/query" in text
        assert "/treatment-card" in text
