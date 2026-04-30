"""
LLM-based metadata enrichment for parsed papers.

Some metadata fields cannot be extracted reliably from PMC XML alone and
benefit from an LLM pass. This module is intentionally narrow — it only runs
a single structured extraction call per paper and returns typed results.

Fields extracted by the LLM:
    study_design: str
        Canonical study type from STUDY_DESIGN_HIERARCHY
        (e.g., "randomised controlled trial", "systematic review").
    patient_population: str
        Brief free-text description of study population
        (e.g., "metastatic CRPC patients, n=246").
    primary_endpoint: str | None
        Primary clinical endpoint if a trial (e.g., "overall survival",
        "PSA response rate").
    intervention: str | None
        Treatment or intervention under investigation.
    comparator: str | None
        Control arm or comparator treatment if present.
    cancer_subtype: str | None
        Fine-grained subtype beyond the broad topic
        (e.g., "non-muscle-invasive bladder cancer", "clear cell RCC").

Implementation notes:
    - Uses `gpt-4o-mini` (METADATA_EXTRACTION_MODEL) with structured output
      (JSON mode / function calling) to guarantee parseable responses.
    - Operates on the abstract + first 500 chars of each section title list;
      does NOT send full body text to keep costs low.
    - Results are cached by pmc_id so re-runs skip already-enriched papers.
    - Failures return a `MetadataExtractionResult` with all fields None and
      `extraction_failed=True`; the paper is still indexed without enrichment.

Public API (to be implemented):
    extract_metadata(paper: ParsedPaper, client: OpenAI) -> MetadataExtractionResult

    MetadataExtractionResult(dataclass)
        pmc_id: str
        study_design: str | None
        patient_population: str | None
        primary_endpoint: str | None
        intervention: str | None
        comparator: str | None
        cancer_subtype: str | None
        extraction_failed: bool
        extraction_model: str
"""
