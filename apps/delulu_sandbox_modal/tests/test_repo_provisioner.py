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

    # --- path-traversal guard ---

    def test_ssh_dotdot_in_host_rejected(self):
        with pytest.raises(ValueError, match="unsafe host"):
            _parse_repo_url("git@../../etc:alice/repo")

    def test_https_dotdot_in_host_rejected(self):
        with pytest.raises(ValueError, match="unsafe host"):
            _parse_repo_url("https://../../:80/alice/repo")

    def test_https_dotdot_in_org_rejected(self):
        with pytest.raises(ValueError, match="unsafe org"):
            _parse_repo_url("https://github.com/../etc/repo")

    def test_https_dotdot_in_repo_rejected(self):
        with pytest.raises(ValueError, match="unsafe repo"):
            _parse_repo_url("https://github.com/alice/../../etc")


class TestBuildAuthHeader:
    """The PAT-as-Basic-Auth header used for /commit pushes."""

    def test_builds_basic_auth_header_with_x_access_token_user(self):
        from delulu_sandbox_modal.repo_provisioner import _build_auth_header

        header = _build_auth_header("ghp_abcdef123456")
        # Basic auth = base64("x-access-token:ghp_abcdef123456")
        assert header.startswith("Authorization: Basic ")
        encoded = header[len("Authorization: Basic ") :]
        import base64

        decoded = base64.b64decode(encoded).decode()
        assert decoded == "x-access-token:ghp_abcdef123456"

    def test_handles_special_characters_in_token(self):
        from delulu_sandbox_modal.repo_provisioner import _build_auth_header

        # Fine-grained PAT format includes underscores and longer
        # length; should encode fine.
        token = "github_pat_11ABCDEFG_xyz123"
        header = _build_auth_header(token)
        import base64

        decoded = base64.b64decode(header[len("Authorization: Basic ") :]).decode()
        assert decoded == f"x-access-token:{token}"


class TestCommitWorkspaceChanges:
    """Pre-flight checks on commit_workspace_changes.

    The full git path is integration-tested by hand against a real
    Modal deploy — the unit test surface is limited to the
    parameter validation branches that don't shell out to git.
    """

    def test_empty_token_raises_value_error(self):
        from delulu_sandbox_modal.repo_provisioner import commit_workspace_changes

        # Defensive check inside the function — caller is supposed
        # to have refused on empty PAT before calling, but we
        # belt-and-suspender it.
        with pytest.raises(ValueError, match="github_token must not be empty"):
            commit_workspace_changes(thread_id=42, message="test", github_token="")

    def test_no_workspace_returns_status(self, tmp_path, monkeypatch):
        """If the workspace dir doesn't exist, return no_workspace cleanly."""
        from delulu_sandbox_modal import repo_provisioner

        # Point WORKSPACES_ROOT at a tmpdir that has no thread
        # subdirs — commit_workspace_changes should detect the
        # missing dir and return a status, not raise.
        monkeypatch.setattr(repo_provisioner, "WORKSPACES_ROOT", str(tmp_path))

        result = repo_provisioner.commit_workspace_changes(
            thread_id=999,
            message="test",
            github_token="ghp_dummy",
        )
        assert result.status == "no_workspace"
        assert "999" in (result.error or "")
