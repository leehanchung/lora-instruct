"""Tests for RepoAllowlist's storage-shape migration logic.

The full ``RepoAllowlist`` class wraps a ``modal.Dict`` and is awkward
to test without mocking the entire Modal client surface, so this file
sticks to the pure-function helper that does the heavy lifting:
``_iter_entries``. It normalizes the three accepted on-disk shapes
(legacy list, current dict, missing) into a uniform
``(owner_repo, visibility)`` projection that every public method
above it consumes. A bug here silently mis-classifies every
allowlist entry on read, which would break the dispatch path's
PAT-required check.
"""

from __future__ import annotations

from delulu_discord.repo_allowlist import (
    VISIBILITY_PRIVATE,
    VISIBILITY_PUBLIC,
    _iter_entries,
)


def test_none_returns_empty():
    assert _iter_entries(None) == []


def test_legacy_list_treats_all_as_public():
    """v1 entries (pre-marker) were stored as a flat list of strings.

    On read they must surface as public — v1 explicitly rejected
    private repos at add time, so anything in a legacy row is
    guaranteed to be a public repo.
    """
    raw = ["alice/api-service", "alice-org/shared-lib"]
    assert _iter_entries(raw) == [
        ("alice/api-service", VISIBILITY_PUBLIC),
        ("alice-org/shared-lib", VISIBILITY_PUBLIC),
    ]


def test_current_dict_extracts_visibility_field():
    raw = {
        "alice/api-service": {"visibility": "public"},
        "alice/internal-svc": {"visibility": "private"},
    }
    entries = dict(_iter_entries(raw))
    assert entries["alice/api-service"] == VISIBILITY_PUBLIC
    assert entries["alice/internal-svc"] == VISIBILITY_PRIVATE


def test_dict_payload_missing_visibility_defaults_public():
    """Forward-compat: a payload dict without a visibility field is treated as public.

    Defaulting to public for unknown payloads is fail-open on the
    visibility marker but fail-closed on the actual auth path —
    a public-classified clone of an actually-private repo just
    fails with the same auth error a missing-marker repo would.
    """
    raw = {"alice/repo": {"some_other_field": "value"}}
    assert _iter_entries(raw) == [("alice/repo", VISIBILITY_PUBLIC)]


def test_dict_non_dict_payload_defaults_public():
    """Defensive: payload that isn't even a dict (e.g. a bare string) -> public."""
    raw = {"alice/repo": "legacy-string-value"}
    assert _iter_entries(raw) == [("alice/repo", VISIBILITY_PUBLIC)]


def test_legacy_list_skips_non_string_entries():
    """Defensive: corrupted dict values get filtered, not crashed on."""
    raw = ["alice/good", 42, None, "bob/also-good"]
    assert _iter_entries(raw) == [
        ("alice/good", VISIBILITY_PUBLIC),
        ("bob/also-good", VISIBILITY_PUBLIC),
    ]


def test_completely_unexpected_shape_returns_empty():
    """Fail-closed on a wholly unexpected stored value rather than crashing."""
    assert _iter_entries(42) == []
    assert _iter_entries("a-bare-string") == []
