"""
FastAPI application package.

Exposes the RAG system as a REST API.  The Streamlit UI (`rag_ui.py`)
calls this API at runtime rather than importing retrieval/generation
modules directly — this decouples the UI from the backend and enables
independent scaling.

Application factory: `api.main.create_app()`.
"""
