# Ingestion Summary

**Date:** 2026-05-05  
**Runs:** 3 (full · incremental · prostate recovery)

---

## Final Corpus

| Metric | Value |
|--------|-------|
| Papers ingested | 27,515 |
| Chunks produced | 685,464 |
| Vector size | 1,536 (text-embedding-3-small) |
| Qdrant collection | `urological_oncology_papers` — GREEN ✅ |

---

## Per-Topic Breakdown

| Topic | Papers | Chunks | Avg Chunks / Paper | Share |
|-------|-------:|-------:|-------------------:|------:|
| Prostate | 15,366 | 387,650 | 25.2 | 56.5% |
| Kidney | 5,244 | 129,176 | 24.6 | 18.8% |
| Bladder | 4,779 | 119,908 | 25.1 | 17.5% |
| Adrenal | 1,185 | 26,801 | 22.6 | 3.9% |
| Testicular | 686 | 16,140 | 23.5 | 2.4% |
| Penile | 255 | 5,789 | 22.7 | 0.8% |
| **Total** | **27,515** | **685,464** | **24.9** | **100%** |

*Prostate numbers derived from total minus other topics (recovered via yearly-window NCBI strategy in run 3).*

---

## Quality Gate

Rejections logged in `data/ingestion_rejected.json` with per-paper score and reasons.
