"""Workspace provisioning — clone repos, manage workspace directories on the volume."""

from __future__ import annotations

import os
import subprocess

import structlog

logger = structlog.get_logger()

WORKSPACES_ROOT = "/vol/workspaces"


def ensure_workspace(session_id: str, repo_url: str | None = None) -> str:
    """Ensure a workspace directory exists, optionally cloning a repo into it.

    Args:
        session_id: Session identifier used as the workspace directory name.
        repo_url: Optional git repo URL to clone. If the workspace already exists
                  and contains files, the clone is skipped.

    Returns:
        Absolute path to the workspace directory.
    """
    workspace_path = os.path.join(WORKSPACES_ROOT, session_id)

    if os.path.exists(workspace_path) and os.listdir(workspace_path):
        logger.info("workspace.exists", path=workspace_path)
        return workspace_path

    os.makedirs(workspace_path, exist_ok=True)

    if repo_url:
        logger.info("workspace.cloning", repo=repo_url, path=workspace_path)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, workspace_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error("workspace.clone_failed", stderr=result.stderr[:500])
            raise RuntimeError(f"Failed to clone {repo_url}: {result.stderr[:200]}")

        logger.info("workspace.cloned", path=workspace_path)
    else:
        logger.info("workspace.created_empty", path=workspace_path)

    return workspace_path


def list_workspaces() -> list[str]:
    """List all workspace session IDs on the volume."""
    if not os.path.exists(WORKSPACES_ROOT):
        return []
    return sorted(os.listdir(WORKSPACES_ROOT))


def workspace_size_mb(session_id: str) -> float:
    """Get approximate size of a workspace in MB."""
    workspace_path = os.path.join(WORKSPACES_ROOT, session_id)
    if not os.path.exists(workspace_path):
        return 0.0

    total = 0
    for dirpath, _dirnames, filenames in os.walk(workspace_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)

    return total / (1024 * 1024)
