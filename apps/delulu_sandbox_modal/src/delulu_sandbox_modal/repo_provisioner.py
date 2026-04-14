"""Git operations for per-thread workspace provisioning.

Three-layer scheme on the Modal Volume:

1.  **Shared bare cache** at ``/vol/repo-cache/<host>/<org>/<repo>.git``.
    Created on first sighting of a URL via
    ``git clone --bare --filter=blob:none``. Partial clone skips blobs
    at clone time, so even large-repo cold clones are small.

2.  **Per-thread worktree** at ``/vol/workspaces/<thread_id>/``, created
    via ``git worktree add``. Shares ``.git/objects`` with the bare
    cache, so disk cost is O(files actually checked out), not
    O(repo-size × thread-count). Stable per thread_id keeps Claude
    Code's ``~/.claude/projects/<hash-of-cwd>/`` session continuity
    intact — ``claude --continue`` keeps working unchanged.

3.  **Refresh policy.** New thread → ensure cache, fetch ref, add
    worktree. Resumed thread → worktree already exists, short-circuit
    via a marker file so we skip all git ops for the warm path.

Everything here is pure Python + ``subprocess`` (no Modal decorators,
no structlog, no bot-side imports). The module sits alongside
``app.py`` but is imported by it at runtime — keeping the decorated
Modal function in ``app.py`` avoids the double-import footgun
documented in that file's module docstring.

All concurrency serialization lives at the Modal layer via
``@app.function(max_containers=1)`` on the wrapping Modal function in
``app.py``. This module deliberately contains **no locking primitives**:
filesystem-based locks don't work on Modal Volumes (see
``apps/delulu_sandbox_modal/tools/verify_flock.py`` for the scout
evidence), and ``max_containers=1`` provides global mutual exclusion
without any lock code at all.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from urllib.parse import urlparse

REPO_CACHE_ROOT = "/vol/repo-cache"
WORKSPACES_ROOT = "/vol/workspaces"

# Written into the per-thread workspace directory after a successful
# provision. On subsequent calls, if the marker records the same
# (repo_url, ref) as the current request, we short-circuit the git
# ops entirely — the worktree is already what the caller wants.
PROVISION_MARKER_NAME = ".provision.json"


@dataclass
class ProvisionTiming:
    """Wall-clock timings for `provision.timing` observability."""

    total_ms: int = 0
    cold_clone_ms: int = 0
    fetch_ms: int = 0
    worktree_ms: int = 0
    short_circuit: bool = False


def provision_workspace(
    thread_id: int,
    repo_url: str | None,
    ref: str = "HEAD",
) -> tuple[str, ProvisionTiming]:
    """Ensure a workspace exists at ``/vol/workspaces/<thread_id>`` and return its path.

    Called from inside the Modal ``provision_workspace`` function in
    ``app.py``, which has ``max_containers=1`` and serializes all
    invocations at the Modal orchestration layer. No lock code is
    needed here because there's only ever one caller active at a time.

    Behavior:

    - ``repo_url is None``: empty workspace for general Q&A mode. No
      git ops, no cache touching. Creates the directory if missing.
    - ``repo_url`` set, worktree missing or stale: ensure the bare
      cache, fetch the ref, create or recreate the worktree, write
      the provision marker.
    - ``repo_url`` set, worktree exists and marker matches: short-
      circuit and return immediately. ``timings.short_circuit = True``.
    """
    start = time.monotonic()
    timings = ProvisionTiming()
    workspace_path = _workspace_path(thread_id)

    if repo_url is None:
        # General Q&A mode — empty workspace, no git.
        os.makedirs(workspace_path, exist_ok=True)
        timings.total_ms = _elapsed_ms(start)
        return workspace_path, timings

    if _workspace_matches(workspace_path, repo_url, ref):
        # Resumed thread on the same repo/ref: short-circuit.
        timings.short_circuit = True
        timings.total_ms = _elapsed_ms(start)
        return workspace_path, timings

    # Ensure the bare cache exists.
    bare_path = _bare_cache_path(repo_url)
    cold_clone = not os.path.isdir(bare_path)
    if cold_clone:
        clone_start = time.monotonic()
        _clone_bare(repo_url, bare_path)
        timings.cold_clone_ms = _elapsed_ms(clone_start)
    else:
        # Warm cache — fetch the ref to pick up upstream changes.
        fetch_start = time.monotonic()
        _fetch_bare(bare_path, ref)
        timings.fetch_ms = _elapsed_ms(fetch_start)

    # Create (or replace) the per-thread worktree.
    worktree_start = time.monotonic()
    _ensure_worktree(bare_path, workspace_path, ref)
    timings.worktree_ms = _elapsed_ms(worktree_start)

    # Record the marker so the next provision on the same thread
    # can short-circuit instead of redoing git work.
    _write_marker(workspace_path, repo_url, ref)

    timings.total_ms = _elapsed_ms(start)
    return workspace_path, timings


def _workspace_path(thread_id: int) -> str:
    return f"{WORKSPACES_ROOT}/{thread_id}"


def _bare_cache_path(repo_url: str) -> str:
    host, org, repo = _parse_repo_url(repo_url)
    return f"{REPO_CACHE_ROOT}/{host}/{org}/{repo}.git"


def _parse_repo_url(repo_url: str) -> tuple[str, str, str]:
    """Return (host, org, repo) for the on-disk cache layout.

    Accepts ``https://github.com/owner/repo[.git]`` and
    ``git@host:owner/repo[.git]`` forms. Raises ``ValueError`` on
    anything it can't parse cleanly — deliberately strict, because a
    bad parse here would silently point at the wrong cache directory.
    """
    url = repo_url.strip()
    if not url:
        raise ValueError("repo_url is empty")

    # Normalize trailing junk. Order matters: strip slash → strip .git
    # → strip slash again, so inputs like
    # ``https://github.com/alice/api-service.git/`` normalize the
    # same as ``.../api-service``.
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    url = url.rstrip("/")

    if url.startswith("git@"):
        # git@host:owner/repo
        rest = url[len("git@") :]
        host, sep, path = rest.partition(":")
        if not sep:
            raise ValueError(f"cannot parse SSH git URL: {repo_url!r}")
    else:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"unsupported URL scheme {parsed.scheme!r} in {repo_url!r} "
                "(only http/https/ssh are accepted)"
            )
        host = parsed.hostname or ""
        path = parsed.path.lstrip("/")

    if not host:
        raise ValueError(f"cannot parse host from {repo_url!r}")

    parts = path.split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"cannot parse owner/repo from {repo_url!r}")

    return host, parts[0], parts[1]


def _clone_bare(repo_url: str, bare_path: str) -> None:
    """``git clone --bare --filter=blob:none <repo_url> <bare_path>``.

    Creates parent directories if missing. Partial clone keeps the
    bare cache small on cold start — file-content blobs are fetched
    on demand when files are actually opened.
    """
    os.makedirs(os.path.dirname(bare_path), exist_ok=True)
    _run_git(
        [
            "clone",
            "--bare",
            "--filter=blob:none",
            repo_url,
            bare_path,
        ]
    )


def _fetch_bare(bare_path: str, ref: str) -> None:
    """``git -C <bare_path> fetch --filter=blob:none origin <ref>``.

    Fetches metadata for the requested ref so ``git worktree add``
    can materialize it. Partial clone keeps this cheap on warm
    caches — usually well under a second.
    """
    _run_git(
        [
            "-C",
            bare_path,
            "fetch",
            "--filter=blob:none",
            "origin",
            ref,
        ]
    )


def _ensure_worktree(bare_path: str, workspace_path: str, ref: str) -> None:
    """Create or reuse a worktree at ``workspace_path`` checked out at ``ref``.

    Handles three cases:

    1.  Workspace doesn't exist → ``git worktree add``.
    2.  Workspace exists and has a ``.git`` file pointing into the bare
        cache → treat as a valid worktree and just ``git checkout`` the
        requested ref. Cheap refresh path.
    3.  Workspace exists but isn't a worktree (e.g. leftover stub from
        an earlier layout) → wipe it and recreate.

    Prunes stale worktree registrations first so the bare cache's
    ``worktrees/`` dir doesn't accumulate dead entries.
    """
    # Prune any dead worktree registrations cheaply.
    _run_git(["-C", bare_path, "worktree", "prune"], check=False)

    if os.path.isdir(workspace_path):
        git_marker = os.path.join(workspace_path, ".git")
        if os.path.exists(git_marker):
            # Valid worktree — just checkout the ref.
            _run_git(["-C", workspace_path, "checkout", ref])
            return
        # Exists but not a worktree. Wipe and recreate below.
        shutil.rmtree(workspace_path)

    os.makedirs(os.path.dirname(workspace_path), exist_ok=True)
    _run_git(
        [
            "-C",
            bare_path,
            "worktree",
            "add",
            "--force",
            workspace_path,
            ref,
        ]
    )


def _workspace_matches(workspace_path: str, repo_url: str, ref: str) -> bool:
    """True if an existing workspace already matches the requested (url, ref).

    Used by the resumed-thread short-circuit in ``provision_workspace``.
    Reads the ``.provision.json`` marker written at the end of a
    successful provision.
    """
    marker_path = os.path.join(workspace_path, PROVISION_MARKER_NAME)
    if not os.path.isfile(marker_path):
        return False
    try:
        with open(marker_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("repo_url") == repo_url and data.get("ref") == ref


def _write_marker(workspace_path: str, repo_url: str, ref: str) -> None:
    """Record the provisioned (repo_url, ref) for next-call short-circuit."""
    marker_path = os.path.join(workspace_path, PROVISION_MARKER_NAME)
    with open(marker_path, "w") as f:
        json.dump({"repo_url": repo_url, "ref": ref}, f)


def _run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Wrapper around ``subprocess.run`` for git commands.

    Captures stderr so callers (and the `provision.timing` logs)
    surface actionable error context on failure. Uses ``text=True``
    so git output is decoded as UTF-8 strings rather than bytes.
    """
    try:
        return subprocess.run(
            ["git", *args],
            check=check,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        # Re-raise with stderr attached to the message — the default
        # CalledProcessError __str__ drops stderr, and debugging a
        # silent git failure against a Modal Volume is painful
        # without it.
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        raise RuntimeError(
            f"git {' '.join(args)} failed with exit code {exc.returncode}: "
            f"stderr={stderr!r} stdout={stdout!r}"
        ) from exc


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
