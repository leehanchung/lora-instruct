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

    # Guard against path-traversal in any component used to build the
    # on-disk cache path.  A URL like ``git@../../etc:alice/repo`` would
    # otherwise resolve outside the volume entirely.
    for label, value in (("host", host), ("org", parts[0]), ("repo", parts[1])):
        if ".." in value or "/" in value:
            raise ValueError(f"unsafe {label} component {value!r} in {repo_url!r}")

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


# ─────────────────────────────────────────────────────────────────
# Commit-back: stage, commit, and push pending changes from a
# per-thread workspace to a `claude/<thread_id>` branch on the
# upstream repo.
# ─────────────────────────────────────────────────────────────────
#
# Lives here (rather than in a separate module) because it's another
# pure-git operation on the same workspaces that ``provision_workspace``
# creates. Same import-isolation rules: no Modal decorators, no bot-
# side imports. The Modal function wrapper that calls this lives in
# ``app.py``.
#
# Refuse-and-instruct UX: callers are expected to check the
# ``GITHUB_TOKEN`` env var BEFORE calling ``commit_workspace_changes``.
# This module doesn't touch the env directly — it just takes the
# token as a parameter so the function is testable without env
# munging and the auth check stays at the boundary.

# The branch name we always commit to. One branch per thread, so a
# user can iterate on the same conceptual "task" across multiple
# /commit calls and have them stack on the same branch — and the
# remote PR (if they open one) shows the full history of the
# thread's changes.
COMMIT_BRANCH_PREFIX = "claude/"

# The git author identity for bot-made commits. The ACTUAL pusher
# (visible in GitHub's UI as "pushed by") is whoever owns the PAT
# stored in the ``github-pat`` Modal secret; this name/email is
# only what shows up in the commit object's Author/Committer
# fields and in ``git log``. Defaults to a generic bot identity
# so commits can be filtered out of git log easily; can be
# overridden via env vars on the github-pat secret (see README).
DEFAULT_GIT_AUTHOR_NAME = "Claude Code"
DEFAULT_GIT_AUTHOR_EMAIL = "claude@bot.local"


@dataclass
class CommitResult:
    """Result of a /commit operation.

    The ``status`` field is the primary signal the bot side renders
    into a Discord response; the other fields carry additional
    context for the success and partial-failure cases.
    """

    status: str  # "ok" | "no_changes" | "no_workspace" | "push_failed"
    branch: str | None = None
    commit_sha: str | None = None
    pr_compare_url: str | None = None
    error: str | None = None


def commit_workspace_changes(
    thread_id: int,
    message: str,
    github_token: str,
    *,
    author_name: str = DEFAULT_GIT_AUTHOR_NAME,
    author_email: str = DEFAULT_GIT_AUTHOR_EMAIL,
) -> CommitResult:
    """Stage, commit, and push pending changes in ``/vol/workspaces/<thread_id>``.

    Idempotent in the no-op sense: if there are no changes, returns
    ``status="no_changes"`` without making a commit. If there ARE
    changes, makes one new commit on the ``claude/<thread_id>``
    branch (creating the branch if missing) and pushes it to
    ``origin``.

    Auth: ``github_token`` is embedded in the push URL as
    ``https://x-access-token:<pat>@host/owner/repo[.git]`` for
    the single ``git push`` invocation. Never persisted to
    ``.git/config``, never written to the remote's stored URL.
    See ``_build_push_url_with_pat`` for the URL construction
    and ``_scrub_pat`` for why error messages get sanitized
    before being surfaced.

    Caller must verify ``github_token`` is non-empty BEFORE calling
    this function — refuse-and-instruct on missing PAT happens at
    the boundary in ``app.py``, not here.
    """
    if not github_token:
        # Defensive — caller should have refused already, but
        # guarding here means a programming error never causes a
        # broken push attempt with empty auth.
        raise ValueError("github_token must not be empty")

    workspace_path = _workspace_path(thread_id)
    if not os.path.isdir(workspace_path):
        return CommitResult(
            status="no_workspace",
            error=f"workspace {workspace_path} does not exist (no prior dispatch in this thread?)",
        )

    # Check for pending changes. ``git status --porcelain`` emits
    # one line per dirty path (modified, added, deleted, untracked);
    # empty output means a clean working tree.
    status_result = _run_git(
        ["-C", workspace_path, "status", "--porcelain"],
    )
    if not status_result.stdout.strip():
        return CommitResult(status="no_changes")

    branch = f"{COMMIT_BRANCH_PREFIX}{thread_id}"

    # Check out the thread's branch. ``-B`` creates if missing OR
    # resets to the current HEAD if existing — but we want to PRESERVE
    # any prior commits on this branch from earlier /commit calls.
    # So: try `git checkout` first; if that fails (branch doesn't
    # exist), `git checkout -b` to create.
    try:
        _run_git(["-C", workspace_path, "checkout", branch])
    except RuntimeError:
        _run_git(["-C", workspace_path, "checkout", "-b", branch])

    # Stage everything and commit with the bot's git identity.
    # ``-c user.name=...`` is per-command, not persisted to config.
    _run_git(["-C", workspace_path, "add", "-A"])
    _run_git(
        [
            "-c",
            f"user.name={author_name}",
            "-c",
            f"user.email={author_email}",
            "-C",
            workspace_path,
            "commit",
            "-m",
            message,
        ]
    )

    # Capture the new commit SHA for the response.
    sha_result = _run_git(["-C", workspace_path, "rev-parse", "HEAD"])
    commit_sha = sha_result.stdout.strip()

    # Push via URL-embedded PAT credentials, passing the full
    # https://x-access-token:<pat>@host/owner/repo[.git] URL as a
    # one-off argument to `git push`. We used to use
    # `-c http.extraheader=Authorization: Basic <base64>` but git
    # would intermittently fall through to the interactive
    # credential prompt in the sandbox, producing:
    #
    #     fatal: could not read Username for 'https://github.com':
    #     No such device or address
    #
    # when git tries to read `/dev/tty` in our non-interactive
    # subprocess. http.extraheader is subtly dependent on the
    # credential helper chain, URL normalization, and whether git
    # considers the hostname scope a match — all of which can vary
    # across git versions and repo configs. URL-embedded
    # credentials are the boring-but-reliable alternative: they
    # bypass git's credential helper machinery entirely because
    # the auth is in the URL itself.
    origin_result = _run_git(
        ["-C", workspace_path, "config", "--get", "remote.origin.url"],
    )
    origin_url = origin_result.stdout.strip()
    push_url = _build_push_url_with_pat(origin_url, github_token)

    try:
        _run_git(["-C", workspace_path, "push", push_url, branch])
    except RuntimeError as exc:
        # Push failed — most commonly an invalid/expired PAT (401)
        # or insufficient scopes. The local commit DID land on the
        # workspace's branch, so the user can re-run /commit after
        # rotating the PAT and the next push will catch up. Surface
        # the git error to the bot so the user sees actionable
        # detail instead of a generic "push failed."
        #
        # **Scrub the PAT** before surfacing: `_run_git`'s error
        # message includes the full args, and one of those args is
        # the push URL with the PAT embedded. Leaving it raw would
        # leak the token into Discord message bodies and bot logs.
        return CommitResult(
            status="push_failed",
            branch=branch,
            commit_sha=commit_sha,
            error=_scrub_pat(str(exc), github_token),
        )

    pr_compare_url = _build_pr_compare_url(workspace_path, branch)

    return CommitResult(
        status="ok",
        branch=branch,
        commit_sha=commit_sha,
        pr_compare_url=pr_compare_url,
    )


def _build_push_url_with_pat(origin_url: str, github_token: str) -> str:
    """Return the origin URL with ``x-access-token:<pat>`` credentials embedded.

    Transforms ``https://github.com/owner/repo[.git]`` into
    ``https://x-access-token:<pat>@github.com/owner/repo[.git]``.

    The PAT becomes part of the URL, which git then uses directly
    for HTTP basic auth without going through the credential
    helper chain. This is the reliable alternative to
    ``-c http.extraheader`` for scripted pushes — git's extraheader
    path has subtle scoping and credential-fallback issues that
    were biting us in the sandbox.

    Only HTTPS URLs are supported. SSH URLs (``git@github.com:…``)
    don't have a natural place to stick a PAT because SSH auth is
    keypair-based, not username/password. The ``/commit`` flow is
    single-shared-PAT-only anyway (see the PRD's v1 scope notes),
    so anything non-HTTPS is rejected loudly.
    """
    from urllib.parse import urlparse, urlunparse

    if not github_token:
        raise ValueError("github_token must not be empty")

    parsed = urlparse(origin_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"cannot push via PAT to non-HTTPS origin {origin_url!r} "
            "(v1 only supports https:// remotes; use a fork or switch "
            "the remote's URL scheme)"
        )

    host = parsed.hostname or ""
    if not host:
        raise ValueError(f"cannot parse host from origin URL {origin_url!r}")

    # Rebuild netloc with ``x-access-token:<pat>@host[:port]``. The
    # ``x-access-token`` username is what GitHub documents for PAT
    # auth over HTTPS — any non-empty string works as the username
    # but this one matches GitHub's docs and makes the intent
    # unambiguous in a packet capture.
    netloc = f"x-access-token:{github_token}@{host}"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    return urlunparse(parsed._replace(netloc=netloc))


def _scrub_pat(message: str, github_token: str) -> str:
    """Replace every occurrence of ``github_token`` in ``message`` with a placeholder.

    Used to sanitize ``git push`` error messages before they hit
    Discord or the bot log. Because we build the push URL with
    the PAT embedded, any git error message that echoes the args
    (which ``_run_git`` does) carries the PAT verbatim — a leak
    straight into whatever renders the error.

    Idempotent and safe to call on empty strings or empty tokens;
    returns the input unchanged in the degenerate cases.
    """
    if not github_token or github_token not in message:
        return message
    return message.replace(github_token, "***PAT***")


def _build_pr_compare_url(workspace_path: str, branch: str) -> str | None:
    """Build a ``github.com/<owner>/<repo>/compare/<base>...<branch>`` link.

    Returns ``None`` if we can't extract the upstream URL or the
    base ref — display code degrades to "branch pushed, open a PR
    yourself" rather than crashing.
    """
    try:
        # Get the origin URL.
        origin_result = _run_git(
            ["-C", workspace_path, "config", "--get", "remote.origin.url"],
        )
        origin_url = origin_result.stdout.strip()
        if not origin_url:
            return None

        # Parse owner/repo out of the origin URL (same logic as the
        # bot side's _short_repo_name in streaming.py, kept inline
        # to avoid cross-app imports).
        host, org, repo = _parse_repo_url(origin_url)
        if host != "github.com":
            # Compare URLs only documented for github.com; skip for
            # other hosts.
            return None

        # Determine the base branch. The worktree was created from a
        # specific ref; we want to compare against that. Read the
        # ref from the .provision.json marker if present.
        marker_path = os.path.join(workspace_path, PROVISION_MARKER_NAME)
        base_ref = "main"
        if os.path.isfile(marker_path):
            try:
                with open(marker_path) as f:
                    base_ref = json.load(f).get("ref") or "main"
            except (OSError, json.JSONDecodeError):
                pass
        if base_ref == "HEAD":
            base_ref = "main"  # GitHub doesn't accept HEAD in compare URLs

        return f"https://github.com/{org}/{repo}/compare/{base_ref}...{branch}?expand=1"
    except (RuntimeError, ValueError):
        return None
