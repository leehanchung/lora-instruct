"""Unit tests for pure-python logic in ``delulu_sandbox_modal.repo_provisioner``.

The git-subprocess helpers (``_clone_bare``, ``_fetch_bare``,
``_ensure_worktree``) aren't tested here — they're thin wrappers
around ``git`` and exercising them requires either a real git
install + temp repo fixture or extensive mocking, neither of which
pays for itself at the current project test-infra level.

What IS worth testing is ``_parse_repo_url``: it's pure, has non-
trivial branching (https vs ssh, trailing .git, owner/repo
validation), and a bug here would silently point at the wrong cache
directory. Cheap insurance.
"""

from __future__ import annotations

import pytest

from delulu_sandbox_modal.repo_provisioner import _parse_repo_url


class TestParseRepoUrl:
    def test_https_github(self):
        assert _parse_repo_url("https://github.com/alice/api-service") == (
            "github.com",
            "alice",
            "api-service",
        )

    def test_https_github_trailing_git(self):
        assert _parse_repo_url("https://github.com/alice/api-service.git") == (
            "github.com",
            "alice",
            "api-service",
        )

    def test_https_github_trailing_slash(self):
        assert _parse_repo_url("https://github.com/alice/api-service/") == (
            "github.com",
            "alice",
            "api-service",
        )

    def test_https_github_trailing_git_and_slash(self):
        # Both normalized away — trailing .git is stripped first, then
        # trailing slash. Order matters.
        assert _parse_repo_url("https://github.com/alice/api-service.git/") == (
            "github.com",
            "alice",
            "api-service",
        )

    def test_https_gitlab(self):
        # Different host — the cache layout slots it under the host dir.
        assert _parse_repo_url("https://gitlab.com/bob/library") == (
            "gitlab.com",
            "bob",
            "library",
        )

    def test_ssh_github(self):
        assert _parse_repo_url("git@github.com:alice/api-service") == (
            "github.com",
            "alice",
            "api-service",
        )

    def test_ssh_github_trailing_git(self):
        assert _parse_repo_url("git@github.com:alice/api-service.git") == (
            "github.com",
            "alice",
            "api-service",
        )

    def test_repo_name_with_dots(self):
        # `.dotfiles` is a real repo name convention — make sure we don't
        # get confused about the .git suffix stripping.
        assert _parse_repo_url("https://github.com/alice/my.weird.repo") == (
            "github.com",
            "alice",
            "my.weird.repo",
        )

    def test_empty_url_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            _parse_repo_url("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            _parse_repo_url("   ")

    def test_missing_owner_rejected(self):
        with pytest.raises(ValueError, match="owner/repo"):
            _parse_repo_url("https://github.com/")

    def test_missing_repo_rejected(self):
        with pytest.raises(ValueError, match="owner/repo"):
            _parse_repo_url("https://github.com/alice")

    def test_ftp_scheme_rejected(self):
        with pytest.raises(ValueError, match="scheme"):
            _parse_repo_url("ftp://github.com/alice/api-service")

    def test_ssh_without_colon_rejected(self):
        with pytest.raises(ValueError, match="SSH git URL"):
            _parse_repo_url("git@github.com/alice/api-service")
