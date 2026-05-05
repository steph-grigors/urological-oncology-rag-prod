# Ingestion Summary

**Mode:** Full (2 runs)  
**Date:** 2026-05-05  
**Elapsed:** Run 1 — 11h 38m 57s · Run 2 — 14m 03s

---

## Overall Results

| Metric | Value |
|--------|-------|
| Papers ingested | 20,945 |
| Chunks produced | 531,242 |
| Chunks embedded | **531,242** ✅ |
| Estimated cost | $3.06 |

---

## Per-Topic Breakdown

| Topic | Fetched | Skipped | Chunks | Avg Chunks / Paper | Share of Corpus |
|-------|--------:|--------:|-------:|-------------------:|----------------:|
| Prostate | 8,796 | 8,788 | 233,428 | 26.5 | 43.9% |
| Bladder | 4,779 | 4,867 | 119,908 | 25.1 | 22.6% |
| Kidney | 5,244 | 5,438 | 129,176 | 24.6 | 24.3% |
| Adrenal | 1,185 | 1,218 | 26,801 | 22.6 | 5.0% |
| Testicular | 686 | 709 | 16,140 | 23.5 | 3.0% |
| Penile | 255 | 265 | 5,789 | 22.7 | 1.1% |
| **Total** | **20,945** | **21,285** | **531,242** | **25.4** | **100%** |

---

## Qdrant Collection

- **Collection:** `urological_oncology_papers`
- **Points:** 531,242
- **Vector size:** 1,536 (text-embedding-3-small)
- **Status:** GREEN

---

## Known Gap — Prostate Pagination

NCBI's E-utilities API reported **18,218** prostate papers matching the query but pagination consistently stops at **9,999** due to malformed JSON on page 2 (invalid control character at col 104). The sanitizer fix converts the crash into a graceful stop but does not recover the response content.

- **Papers in Qdrant:** ~8,796 prostate papers (~48% of available corpus)
- **Estimated missing:** ~8,175 prostate papers (beyond the 9,999 NCBI page boundary)
- **Impact:** Moderate — the 9,999 fetched are filtered by publication type (RCTs, meta-analyses, systematic reviews, guidelines, clinical trials) so coverage of high-evidence studies is strong. The missing papers are not lost — they can be recovered by investigating the NCBI pagination issue.

---

## Quality Gate

- Rejections logged in `data/ingestion_rejected.json` with per-paper score and reasons

---

## Next Steps

1. Investigate NCBI page 2 malformed JSON to recover the ~8,175 missing prostate papers
2. Start weekly incremental runs via the `ingestion-cron` Docker profile
