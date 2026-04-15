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


class TestBuildPushUrlWithPat:
    """URL-embedded PAT credentials for the /commit push path.

    The token is placed in the **password** field of the Basic auth
    component, with ``git`` as a placeholder username:

        https://git:<pat>@github.com/owner/repo

    See ``_build_push_url_with_pat``'s docstring for the full
    four-attempt saga. The test_no_xaccesstoken_regression and
    test_pat_in_password_not_username_field tests guard against
    the two failure modes we've already hit.
    """

    def test_https_github_url_gets_credentials_embedded(self):
        from delulu_sandbox_modal.repo_provisioner import _build_push_url_with_pat

        url = _build_push_url_with_pat(
            "https://github.com/alice/api-service.git",
            "ghp_abcdef123456",
        )
        assert url == "https://git:ghp_abcdef123456@github.com/alice/api-service.git"

    def test_https_github_url_without_dot_git_suffix(self):
        from delulu_sandbox_modal.repo_provisioner import _build_push_url_with_pat

        url = _build_push_url_with_pat(
            "https://github.com/alice/api-service",
            "ghp_abcdef123456",
        )
        assert url == "https://git:ghp_abcdef123456@github.com/alice/api-service"

    def test_pat_in_password_not_username_field(self):
        """Regression guard — PR #58's mistake was putting the PAT alone in userinfo.

        GitHub rejected token-only userinfo (``<pat>@github.com``)
        because it wants the PAT in the password field. Git then
        tried to read a password interactively and failed with
        "could not read Password: No such device or address."

        The userinfo must contain a ``:`` splitting a non-empty
        username from the PAT, and the PAT must be AFTER the
        colon (the password field).
        """
        from delulu_sandbox_modal.repo_provisioner import _build_push_url_with_pat

        token = "ghp_abcdef"
        url = _build_push_url_with_pat("https://github.com/alice/api-service", token)
        # Extract the userinfo component
        assert "://" in url and "@" in url
        userinfo = url.split("://", 1)[1].split("@", 1)[0]
        # Must have a colon splitting username:password
        assert ":" in userinfo, f"userinfo must be 'user:pat' form, got {userinfo!r}"
        username, _, password = userinfo.partition(":")
        assert username, "username field must be non-empty"
        assert password == token, f"PAT must be in password field, not username; got {userinfo!r}"

    def test_no_xaccesstoken_regression(self):
        """Regression guard — PR #56's mistake was ``x-access-token:<pat>``.

        The ``x-access-token`` username is for GitHub App installation
        tokens, not Personal Access Tokens. GitHub's auth layer routes
        it to the App-token handler, which rejects PATs with a
        misleading "password authentication not supported" error. This
        test fails loudly if anyone ever re-adds that username.
        """
        from delulu_sandbox_modal.repo_provisioner import _build_push_url_with_pat

        url = _build_push_url_with_pat(
            "https://github.com/alice/api-service",
            "ghp_abcdef",
        )
        assert "x-access-token" not in url

    def test_fine_grained_pat_format_roundtrips(self):
        """Fine-grained PATs have underscores and longer length — should be fine."""
        from delulu_sandbox_modal.repo_provisioner import _build_push_url_with_pat

        token = "github_pat_11ABCDEFG_xyz123"
        url = _build_push_url_with_pat("https://github.com/alice/api-service", token)
        assert f":{token}@github.com" in url

    def test_non_default_port_preserved(self):
        from delulu_sandbox_modal.repo_provisioner import _build_push_url_with_pat

        url = _build_push_url_with_pat(
            "https://github.example.com:8443/alice/api-service.git",
            "ghp_abcdef",
        )
        assert "git:ghp_abcdef@github.example.com:8443" in url

    def test_ssh_url_rejected(self):
        """v1 only supports HTTPS remotes; SSH urls have no place for a PAT."""
        from delulu_sandbox_modal.repo_provisioner import _build_push_url_with_pat

        with pytest.raises(ValueError, match="non-HTTPS origin"):
            _build_push_url_with_pat(
                "git@github.com:alice/api-service.git",
                "ghp_abcdef",
            )

    def test_empty_token_rejected(self):
        from delulu_sandbox_modal.repo_provisioner import _build_push_url_with_pat

        with pytest.raises(ValueError, match="github_token must not be empty"):
            _build_push_url_with_pat(
                "https://github.com/alice/api-service.git",
                "",
            )


class TestScrubPat:
    """Sanitize git error messages so the PAT doesn't leak to Discord/logs."""

    def test_scrubs_pat_from_message(self):
        from delulu_sandbox_modal.repo_provisioner import _scrub_pat

        token = "ghp_abcdef123456"
        msg = (
            "git -C /vol/workspaces/42 push "
            "https://x-access-token:ghp_abcdef123456@github.com/alice/api-service.git "
            "claude/42 failed with exit code 128: stderr='fatal: auth failed'"
        )
        scrubbed = _scrub_pat(msg, token)
        assert token not in scrubbed
        assert "***PAT***" in scrubbed

    def test_handles_multiple_occurrences(self):
        from delulu_sandbox_modal.repo_provisioner import _scrub_pat

        token = "ghp_xyz"
        msg = f"URL1: {token} URL2: {token}"
        scrubbed = _scrub_pat(msg, token)
        assert token not in scrubbed
        assert scrubbed.count("***PAT***") == 2

    def test_empty_token_leaves_message_unchanged(self):
        from delulu_sandbox_modal.repo_provisioner import _scrub_pat

        msg = "some error message"
        assert _scrub_pat(msg, "") == msg

    def test_missing_token_leaves_message_unchanged(self):
        """Defensive: if the token isn't in the message, don't touch it."""
        from delulu_sandbox_modal.repo_provisioner import _scrub_pat

        assert _scrub_pat("no secrets here", "ghp_xxx") == "no secrets here"


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
