"""
Root pytest conftest.

Two jobs, both about making a test run depend on nothing but the repository:

  1. Put the project root on sys.path so `from src.ingestion.chunk import ...`
     resolves in every test module. Run tests from the project root: pytest
  2. Seed the one piece of configuration the suite cannot start without.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Hermetic configuration ───────────────────────────────────────────────────
# config.settings.Settings declares OPENAI_API_KEY as a required field. On a
# developer machine that requirement is quietly satisfied by the gitignored
# .env file, so the suite passed locally while failing anywhere that file does
# not exist -- CI, a fresh clone, a build container. Nine tests that exercise
# src/ingestion/pipeline.run_ingestion (which constructs Settings directly)
# failed exactly that way.
#
# Seed a placeholder here instead. This must happen before anything imports
# config.settings, because get_settings() is lru_cached and the first
# construction wins; a root conftest is loaded before any test module, so this
# is the earliest available hook.
#
# Assigning rather than using setdefault is deliberate: environment variables
# take precedence over .env in pydantic-settings, so this also guarantees that
# no test can reach a live API with a developer's real key. Nothing in the
# suite makes a real network call -- every external client is mocked or
# in-memory -- so a placeholder is sufficient.
os.environ["OPENAI_API_KEY"] = "sk-test-placeholder-not-a-real-key"
