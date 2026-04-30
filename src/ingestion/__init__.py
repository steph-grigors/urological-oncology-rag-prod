"""
Ingestion package.

Responsible for the full data pipeline from raw PMC XML to indexed,
embedded chunks ready for retrieval:

    fetch → parse → chunk → extract_metadata → embed → (index in Qdrant)

Entry point: `pipeline.run_ingestion(topics, output_dir)`.
"""
