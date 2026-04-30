"""
GET /health — liveness and readiness probes.

Endpoints:
    GET /health/live
        Always returns 200 {"status": "ok"} if the process is running.
        Used by Docker / Kubernetes liveness probe.

    GET /health/ready
        Returns 200 only if all dependencies are reachable:
            - Qdrant: collection.count() responds within 2s
            - Postgres: SELECT 1 responds within 2s
            - OpenAI embedding: cached no-op ping (not a real API call)
        Returns 503 with a body indicating which dependency is down.
        Used by Kubernetes readiness probe to gate traffic.

    GET /health/info
        Returns system metadata:
            app_env, generation_model, embedding_model,
            collection_name, collection_count, db_version,
            uptime_seconds
        Protected by API key auth (same as /query).
"""
