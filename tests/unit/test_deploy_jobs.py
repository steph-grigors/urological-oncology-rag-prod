"""
Tests for the scheduled-job wiring in docker/.

Both jobs in the ingestion-cron profile were broken in ways nothing detected,
because neither the Dockerfile nor the compose labels were covered by a test:

  - the regulatory refresh ran `python scripts/update_regulatory_db.py` inside
    a container whose image never contained scripts/
  - the weekly ingestion passed `--since-date $(date -d '7 days ago' ...)`,
    but ofelia runs a job's command directly rather than through a shell, so
    the substitution was never evaluated

These tests read the real Dockerfile and compose file, so they fail if either
regresses.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"


# ── --since-days ─────────────────────────────────────────────────────────────

class TestSinceDaysResolution:
    def test_resolves_to_a_pubmed_style_date(self):
        from src.ingestion.pipeline import _since_date_from_days

        expected = (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y/%m/%d")
        assert _since_date_from_days(7) == expected

    @pytest.mark.parametrize("value", [None, 0, -3])
    def test_non_positive_means_no_restriction(self, value):
        from src.ingestion.pipeline import _since_date_from_days

        assert _since_date_from_days(value) is None

    def test_cli_accepts_the_flag(self):
        from src.ingestion.pipeline import _build_parser

        args = _build_parser().parse_args(["--since-days", "7"])
        assert args.since_days == 7

    def test_explicit_since_date_still_parses(self):
        from src.ingestion.pipeline import _build_parser

        args = _build_parser().parse_args(["--since-date", "2026/01/01"])
        assert args.since_date == "2026/01/01"
        assert args.since_days is None


# ── Image contents ───────────────────────────────────────────────────────────

class TestImageContainsWhatRuntimeReads:
    """Every path the container reads at runtime has to be COPYed in. The
    test suite itself deliberately stays out of the image -- only the one
    golden-set fixture the eval route needs is copied."""

    def test_scripts_are_copied(self):
        assert re.search(r"^COPY scripts/", DOCKERFILE.read_text(), re.M), (
            "scripts/ missing from the image; the weekly regulatory refresh "
            "job runs python scripts/update_regulatory_db.py inside it"
        )

    def test_golden_set_is_copied(self):
        from src.evaluation.runner import run_evaluation
        import inspect

        default_path = inspect.signature(run_evaluation).parameters["golden_set_path"].default
        assert default_path in DOCKERFILE.read_text(), (
            f"{default_path} is read at runtime by POST /eval/run but is not "
            "in the image"
        )

    def test_the_test_suite_is_not_shipped(self):
        text = DOCKERFILE.read_text()
        assert not re.search(r"^COPY tests/\s", text, re.M)
        assert not re.search(r"^COPY tests/ ", text, re.M)


# ── Cron commands ────────────────────────────────────────────────────────────

class TestCronCommandsNeedNoShell:
    """ofelia executes a job's command directly. Anything requiring shell
    evaluation is passed through literally and silently does the wrong thing."""

    def _job_commands(self) -> dict[str, str]:
        """Commands from both services. The two api jobs are job-exec labels on
        the target container; the restart job is a job-run on ofelia itself."""
        import yaml

        compose = yaml.safe_load(COMPOSE.read_text())
        out = {}
        for service in ("api", "ingestion-cron"):
            labels = compose["services"][service].get("labels", {}) or {}
            out.update({k: v for k, v in labels.items() if k.endswith(".command")})
        return out

    def test_no_command_substitution_anywhere(self):
        for name, command in self._job_commands().items():
            assert "$(" not in command, f"{name} relies on shell substitution: {command!r}"
            assert "`" not in command, f"{name} relies on shell substitution: {command!r}"

    def test_weekly_ingestion_uses_since_days(self):
        commands = self._job_commands()
        weekly = commands["ofelia.job-exec.weekly-ingestion.command"]
        assert "--since-days" in weekly
        assert "--since-date" not in weekly

    def test_weekly_ingestion_flags_are_all_accepted_by_the_cli(self):
        """The strongest check available without running ofelia: take the
        command the scheduler will actually issue and parse it with the real
        argument parser."""
        from src.ingestion.pipeline import _build_parser

        weekly = str(self._job_commands()["ofelia.job-exec.weekly-ingestion.command"])
        # The job now chains ingestion and a cache rebuild inside sh -c, so
        # pull out just the pipeline invocation and parse that.
        import re as _re

        m = _re.search(r"python -m src\.ingestion\.pipeline([^&\"]*)", weekly)
        assert m, f"no pipeline invocation found in {weekly!r}"
        args = _build_parser().parse_args(m.group(1).split())
        assert args.since_days == 7
        assert args.limit == 2000


# ── Build context ────────────────────────────────────────────────────────────

def _copy_sources() -> list[str]:
    """Paths the Dockerfile copies from the build context (not --from stages)."""
    return re.findall(r"^COPY (?!--from)(\S+)", DOCKERFILE.read_text(), re.M)


def _dockerignore_patterns() -> tuple[list[str], set[str]]:
    lines = [l.strip() for l in DOCKERIGNORE.read_text().splitlines()
             if l.strip() and not l.strip().startswith("#")]
    negations = {p[1:] for p in lines if p.startswith("!")}
    return [p for p in lines if not p.startswith("!")], negations


class TestBuildContext:
    """.dockerignore and the Dockerfile have to agree. Excluding something the
    Dockerfile copies produces a build failure; excluding too little ships the
    whole repository, which here meant a 5.1 GB Qdrant snapshot and both .env
    files travelling to the daemon on every build."""

    def test_no_copy_source_is_excluded(self):
        import fnmatch

        patterns, negations = _dockerignore_patterns()

        def ignored(path: str) -> str | None:
            if path in negations:
                return None
            for pattern in patterns:
                probe = pattern.rstrip("/")
                if (fnmatch.fnmatch(path, pattern)
                        or fnmatch.fnmatch(path, probe)
                        or path.startswith(probe + "/")):
                    return pattern
            return None

        excluded = {src: ignored(src.rstrip("/")) for src in _copy_sources()}
        offenders = {k: v for k, v in excluded.items() if v}
        assert not offenders, f"Dockerfile COPY sources excluded by .dockerignore: {offenders}"

    @pytest.mark.parametrize("pattern", ["*.snapshot", ".env", "data/"])
    def test_heavy_or_secret_paths_are_excluded(self, pattern):
        patterns, _ = _dockerignore_patterns()
        assert pattern in patterns, f"{pattern} should not be in the build context"


# ── Scheduler registration ───────────────────────────────────────────────────

class TestOfeliaJobsCanRegister:
    """The ingestion-cron profile had never been started in production. When it
    first was, on 2026-09-21, every job failed to register and ofelia
    crash-looped with "unable to start a empty scheduler". Verified against the
    real image: a 5-field schedule is rejected, a 6-field one registers."""

    def _labels(self, service: str) -> dict:
        import yaml

        compose = yaml.safe_load(COMPOSE.read_text())
        return compose["services"][service].get("labels", {}) or {}

    def _schedules(self) -> dict[str, str]:
        out = {}
        for service in ("api", "ingestion-cron"):
            for key, value in self._labels(service).items():
                if key.endswith(".schedule"):
                    out[key] = str(value).strip()
        return out

    def test_there_are_scheduled_jobs_at_all(self):
        assert self._schedules(), "no ofelia schedules defined"

    @pytest.mark.parametrize("expected", ["regulatory-db-update", "weekly-ingestion", "restart-api"])
    def test_each_job_is_present(self, expected):
        assert any(expected in k for k in self._schedules()), f"{expected} job missing"

    def test_every_schedule_is_six_field_cron(self):
        """ofelia leads with seconds. A 5-field schedule registers as nothing
        and the whole scheduler exits, taking the other jobs with it."""
        for key, schedule in self._schedules().items():
            if schedule.startswith("@"):
                continue
            fields = schedule.split()
            assert len(fields) == 6, (
                f"{key} has {len(fields)} cron fields ({schedule!r}); "
                "ofelia requires 6, leading with seconds"
            )

    def test_exec_jobs_live_on_the_target_container(self):
        """job-exec labels are read from the container the command runs in,
        which must also carry ofelia.enabled."""
        api = self._labels("api")
        assert str(api.get("ofelia.enabled", "")).lower() == "true"
        assert any(k.startswith("ofelia.job-exec.") for k in api)
        cron = self._labels("ingestion-cron")
        assert not any(k.startswith("ofelia.job-exec.") for k in cron), (
            "job-exec labels on the ofelia container itself do nothing"
        )

    def test_no_job_depends_on_a_binary_the_ofelia_image_lacks(self):
        """The original restart job called curl against the Docker socket. The
        ofelia image has neither curl nor the Docker CLI, so it would have
        failed at runtime even once registered."""
        for key, value in self._labels("ingestion-cron").items():
            if not key.endswith(".command"):
                continue
            job = key.rsplit(".", 1)[0]
            if f"{job}.image" in self._labels("ingestion-cron"):
                continue  # runs in its own image, which supplies its own tools
            for binary in ("curl", "docker"):
                assert binary not in str(value), (
                    f"{key} uses {binary}, absent from the ofelia image"
                )

    def test_ingestion_rebuilds_the_bm25_cache(self):
        """The cache is keyed on the collection point count, so ingesting makes
        it stale. Without a rebuild the restart re-scrolls the whole collection
        inline and takes minutes instead of seconds."""
        api = self._labels("api")
        command = str(api["ofelia.job-exec.weekly-ingestion.command"])
        assert "src.ingestion.pipeline" in command
        assert "rebuild_bm25_cache" in command

    def test_the_rebuild_script_exists_and_is_in_the_image(self):
        assert (REPO_ROOT / "scripts" / "rebuild_bm25_cache.py").is_file()
        assert re.search(r"^COPY scripts/", DOCKERFILE.read_text(), re.M)
