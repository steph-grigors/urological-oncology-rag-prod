# Ingestion Summary

**Date:** 2026-05-10  
**Runs:** 4 (full · incremental · prostate recovery · 2025–2026 catch-up)

---

## Final Corpus

| Metric | Value |
|--------|-------|
| Papers ingested | 31,361 |
| Chunks produced | 795,306 |
| Vector size | 1,536 (text-embedding-3-small) |
| Qdrant collection | `urological_oncology_papers` — GREEN ✅ |

---

## Per-Topic Breakdown

| Topic | Papers | Chunks | Avg Chunks / Paper | Share |
|-------|-------:|-------:|-------------------:|------:|
| Prostate | 17,382 | 445,895 | 25.6 | 56.1% |
| Kidney | 6,034 | 152,113 | 25.2 | 19.1% |
| Bladder | 5,476 | 139,933 | 25.6 | 17.6% |
| Adrenal | 1,384 | 31,774 | 23.0 | 4.0% |
| Testicular | 782 | 18,479 | 23.6 | 2.3% |
| Penile | 303 | 7,112 | 23.5 | 0.9% |
| **Total** | **31,361** | **795,306** | **25.4** | **100%** |

---

## Run 4 — 2025–2026 catch-up (2026-05-10)

| Metric | Value |
|--------|-------|
| Papers fetched | 3,846 |
| Chunks produced | 109,842 |
| Embedded | 109,842 |
| Cost | $0.6345 |
| Elapsed | 3,448.6s (~57 min) |

---

## Quality Gate

Rejections logged in `data/ingestion_rejected.json` with per-paper score and reasons.
