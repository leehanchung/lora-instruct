"""Integration tests for repo provisioning and the commit-back flow.

Tests that the sandbox can:
  - Clone a public repo into the bare cache
  - Create a worktree at the expected workspace path
  - Short-circuit on repeated provisioning (same thread + repo)
  - Run Claude Code against a provisioned workspace
  - Stage, commit, and push changes (commit-back)
"""

from __future__ import annotations

import modal
import pytest

from .conftest import TEST_REF, TEST_REPO_URL


class TestProvisionWorkspace:
    """Test the ``provision_workspace`` Modal function directly."""

    def test_provision_returns_workspace_path(
        self,
        provision_workspace_fn: modal.Function,
        unique_thread_id: int,
    ) -> None:
        """Provisioning a repo should return the workspace path string."""
        workspace_path = provision_workspace_fn.remote(
            thread_id=unique_thread_id,
            repo_url=TEST_REPO_URL,
            ref=TEST_REF,
        )

        assert isinstance(workspace_path, str)
        assert str(unique_thread_id) in workspace_path
        assert workspace_path.startswith("/vol/workspaces/")

    def test_provision_short_circuits_on_repeat(
        self,
        provision_workspace_fn: modal.Function,
        unique_thread_id: int,
    ) -> None:
        """Second call with same (thread_id, repo_url, ref) should short-circuit.

        We can't directly observe the short-circuit from the return
        value (it returns the same path either way), but we can
        verify it doesn't error and returns quickly.
        """
        # First call — cold clone
        path1 = provision_workspace_fn.remote(
            thread_id=unique_thread_id,
            repo_url=TEST_REPO_URL,
            ref=TEST_REF,
        )

        # Second call — should short-circuit via .provision.json marker
        path2 = provision_workspace_fn.remote(
            thread_id=unique_thread_id,
            repo_url=TEST_REPO_URL,
            ref=TEST_REF,
        )

        assert path1 == path2

    def test_provision_no_repo_returns_workspace_path(
        self,
        provision_workspace_fn: modal.Function,
        unique_thread_id: int,
    ) -> None:
        """Provisioning with no repo should still return a valid workspace path."""
        workspace_path = provision_workspace_fn.remote(
            thread_id=unique_thread_id,
            repo_url=None,
        )

        assert isinstance(workspace_path, str)
        assert str(unique_thread_id) in workspace_path


class TestRunWithRepo:
    """Test ``run_claude_code`` with a repo binding."""

    def test_claude_can_read_repo_files(
        self,
        run_claude_code_fn: modal.Function,
        unique_thread_id: int,
    ) -> None:
        """Claude Code should be able to see and read files from the provisioned repo."""
        events = list(
            run_claude_code_fn.remote_gen(
                session_id="repo-read-files",
                prompt=(
                    "List the files in the current directory. "
                    "Reply with just the filenames, one per line."
                ),
                thread_id=unique_thread_id,
                repo_url=TEST_REPO_URL,
                ref=TEST_REF,
            )
        )

        done = next((e for e in events if e["type"] == "done"), None)
        if done is None:
            error = next((e for e in events if e["type"] == "error"), None)
            pytest.fail(f"Expected done event, got error: {error}")

        # The SMILE-factory repo should have at least a README or pyproject.toml
        output = done["final_text"].lower()
        assert any(name in output for name in ["readme", "pyproject", "claude.md", "makefile"]), (
            f"Expected repo files in output, got: {done['final_text'][:300]}"
        )


class TestCommitWorkspace:
    """Test the ``commit_workspace`` Modal function."""

    def test_commit_no_workspace_returns_status(
        self,
        commit_workspace_fn: modal.Function,
        unique_thread_id: int,
    ) -> None:
        """Committing a non-existent workspace should return no_workspace status."""
        result = commit_workspace_fn.remote(
            thread_id=unique_thread_id,
            message="test commit",
        )

        assert isinstance(result, dict)
        assert result["status"] in ("no_workspace", "no_pat")

    def test_commit_no_changes_returns_status(
        self,
        run_claude_code_fn: modal.Function,
        commit_workspace_fn: modal.Function,
        unique_thread_id: int,
    ) -> None:
        """Committing a clean workspace (no modifications) should return no_changes."""
        # Provision the workspace by running a trivial prompt with a repo
        list(
            run_claude_code_fn.remote_gen(
                session_id="commit-clean",
                prompt="What is 1+1? Reply with just the number.",
                thread_id=unique_thread_id,
                repo_url=TEST_REPO_URL,
                ref=TEST_REF,
            )
        )

        result = commit_workspace_fn.remote(
            thread_id=unique_thread_id,
            message="test commit on clean workspace",
        )

        assert isinstance(result, dict)
        # Should be no_changes (clean workspace) or no_pat (placeholder token)
        assert result["status"] in ("no_changes", "no_pat"), (
            f"Expected no_changes or no_pat, got: {result}"
        )
