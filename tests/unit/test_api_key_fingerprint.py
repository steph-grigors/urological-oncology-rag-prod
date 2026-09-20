"""
Unit tests for src.api.middleware.auth.api_key_fingerprint.

The audit log records which caller ran each query. It used to record the API
key itself, which made `audit_log.user_id` a credential store: read access to
the audit table became API access. The fingerprint keeps attribution without
that property.
"""

from __future__ import annotations

from src.api.middleware.auth import api_key_fingerprint


class TestApiKeyFingerprint:
    def test_returns_none_for_unauthenticated_sentinel(self):
        assert api_key_fingerprint("") is None

    def test_returns_none_for_development_sentinel(self):
        assert api_key_fingerprint("dev") is None

    def test_does_not_contain_the_key(self):
        key = "sk-live-super-secret-value"
        fp = api_key_fingerprint(key)
        assert fp is not None
        assert key not in fp
        assert "secret" not in fp

    def test_is_stable_across_calls(self):
        """Attribution and grouping both depend on the same key always
        mapping to the same fingerprint."""
        assert api_key_fingerprint("key-abc") == api_key_fingerprint("key-abc")

    def test_distinguishes_different_keys(self):
        assert api_key_fingerprint("key-abc") != api_key_fingerprint("key-xyz")

    def test_fits_the_user_id_column(self):
        """audit_log.user_id is String(100)."""
        fp = api_key_fingerprint("k" * 500)
        assert fp is not None
        assert len(fp) <= 100

    def test_is_prefixed_so_stored_values_are_recognisable(self):
        fp = api_key_fingerprint("key-abc")
        assert fp is not None
        assert fp.startswith("k_")
