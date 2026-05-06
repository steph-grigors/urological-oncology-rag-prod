# Tests Information

## Test Suites

There are three independent test suites. None require live APIs or Docker.

---

## 1. Unit Tests — `tests/unit/`

**130 tests, ~8s, no external dependencies.**

Test the individual components in isolation with mocked dependencies.

```bash
python -m pytest tests/unit/ -v
```

| File | What it tests |
|---|---|
| `test_chunking.py` | Section-aware chunker, chunk sizes, overlap, metadata propagation |
| `test_confidence_gating.py` | Confidence gate routing: high / hedged / refuse thresholds |
| `test_hybrid_search.py` | RRF fusion, score normalisation, deduplication |
| `test_metadata_extraction.py` | GPT-4o-mini metadata extractor, caching, failure handling |
| `test_paper_quality.py` | Quality gate scoring, pass/fail thresholds, rejection logging |

---

## 2. Integration Tests — `tests/integration/`

**31 tests, ~3s, uses in-memory Qdrant (no Docker needed).**

Tests the full retrieval + generation pipeline end-to-end using:
- 10 JATS XML fixture papers in `tests/fixtures/sample_papers/`
- Deterministic fake embeddings (90% cancer-type basis + 10% noise — no OpenAI call)
- Mocked LLM responses (no GPT-4o-mini call)
- In-memory Qdrant (no Docker)
- In-process SQLite for audit logs

```bash
python -m pytest tests/integration/ -v
```

| File | What it tests |
|---|---|
| `test_retrieval_pipeline.py` | Dense search, BM25, hybrid fusion, cancer-type filtering, reranker |
| `test_generation_pipeline.py` | ClinicalGenerator, confidence gate, FastAPI `/query` route, auth, rate limiting, streaming |

---

## 3. Evaluation / Golden Set — `tests/eval/`

**Quality regression gate. Run before merging pipeline changes.**

Runs the full RAG against a curated set of real clinical questions and asserts
quality metrics stay above defined floors.

```bash
python -m pytest tests/eval/ -m eval -v
```

**Quality floors:**

| Metric | Minimum |
|---|---|
| Faithfulness | ≥ 0.90 |
| Answer relevance | ≥ 0.85 |
| Context precision | ≥ 0.80 |
| Clinical safety | ≥ 0.99 |
| P95 latency | ≤ 8,000ms |

Golden queries are defined in `tests/fixtures/golden_queries.json`.

---

## Running Everything

```bash
# All unit + integration (fast, no APIs needed)
python -m pytest tests/unit/ tests/integration/ -v

# Full suite including eval gate (requires live OpenAI + Qdrant)
python -m pytest tests/ -v

# Specific marker only
python -m pytest -m integration -v
python -m pytest -m eval -v
```

---

## Current Status

| Suite | Tests | Result |
|---|---|---|
| Unit | 130 | ✅ All passing |
| Integration | 31 | ✅ All passing |
| Eval / golden set | — | Requires live stack |

---

## Test Infrastructure

- **Fixtures:** `tests/fixtures/sample_papers/` — 10 real JATS XML papers (PMC1001–PMC1010)
- **Golden set:** `tests/fixtures/golden_queries.json` — curated clinical Q&A pairs
- **conftest.py:** Session-scoped in-memory Qdrant + BM25 index shared across integration tests
- **pytest.ini:** Defines `integration` and `eval` markers
