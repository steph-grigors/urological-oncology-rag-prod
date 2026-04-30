"""
Retrieval package.

Implements a hybrid retrieval pipeline:

    dense (Qdrant ANN)  ─┐
                          ├── RRF fusion ──► cross-encoder rerank ──► top-k chunks
    sparse (BM25)       ─┘

Entry point: `retriever.Retriever` — the only class the generation layer
and API routes should import.
"""
