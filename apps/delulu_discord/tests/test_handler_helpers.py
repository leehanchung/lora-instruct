"""Pure-function helper tests from delulu_discord.handlers.

The handler classes themselves require a discord.py message/thread
fixture to test, but the URL-parsing helper is pure and load-bearing
for the dispatch path's allowlist visibility lookup — a bug here
silently misses the marker and the dispatch falls back to public,
which means private repos with no PAT would surface as opaque clone
failures instead of refuse-and-instruct.
"""

from __future__ import annotations

from delulu_discord.handlers import _short_repo_from_url


def test_short_repo_from_url_https():
    assert _short_repo_from_url("https://github.com/alice/api-service") == "alice/api-service"


def test_short_repo_from_url_https_trailing_git():
    assert (
        _short_repo_from_url("https://github.com/alice/api-service.git") == "alice/api-service"
    )


def test_short_repo_from_url_ssh():
    assert (
        _short_repo_from_url("git@github.com:alice/api-service.git") == "alice/api-service"
    )


def test_short_repo_from_url_unparseable_returns_none():
    """Unlike streaming._short_repo_name, this helper returns None on bad input.

    The caller uses the result as an allowlist key — querying for
    a nonsense entry pollutes the lookup and might surface as a
    weird visibility result. None lets the caller skip the lookup
    entirely.
    """
    assert _short_repo_from_url("not-a-url") is None
    assert _short_repo_from_url("") is None
