# Efficient GitHub repo provisioning for discord-orchestrator

## Context

The discord-orchestrator bot is an @mention-gated interface to Claude Code
running in ephemeral Modal sandboxes. Today the sandbox function creates an
empty workspace directory and runs `claude -p <prompt>` in it — there's no
repo, so Claude can only answer general questions. The user wants to extend
this to code-editing tasks on GitHub repos, and is specifically worried
about **cold-start latency**: a naive `git clone` per invocation on a
medium repo is 10–60s, which makes the bot feel dead.

An earlier pass at repo provisioning lived in a now-deleted
`workspace.py` with a `git clone --depth 1` implementation, never wired
up. Its mechanism (full clone per session, no refresh) didn't solve the
cold-start problem and is being replaced by the scheme below.

## Goals / non-goals

**Goals**
- Any thread can edit a checked-out repo, with provisioning in ≤ a few
  seconds on the warm path (repo seen before) and ≤ ~10s cold (first time
  on this repo).
- Session continuity via `claude --continue` keeps working — the workspace
  path must stay stable per `thread_id`.
- Public GitHub repos work without any secret setup.
- Optional commit-back via `/commit` slash command when the user has
  configured a PAT.

**Non-goals (v1)**
- Private repo clones (deferred — requires PAT + auth rewriting).
- Auto-PR creation.
- Git submodules, Git LFS.
- Warm-pool Modal containers (`min_containers=1`) — user opted out.
- Workspace GC (volumes persist indefinitely for now).

## User decisions locked in

| Question | Choice |
|---|---|
| Repo specification UX | Per-channel binding via `/setrepo` slash command |
| Git auth | Public repos only for clone; optional PAT for `/commit` push |
| Warm container pool | No — cold container start is acceptable |
| v1 scope | Provisioning + `/commit` commit-back |

### Note on the auth / commit-back tension

"Public repos only" means clone/fetch need no credentials. But `/commit`
requires *push*, which GitHub requires auth for even on public repos.
Resolution: clone path uses anonymous HTTPS; `/commit` path reads an
optional `github-pat` Modal secret and errors cleanly if it's absent
(`"configure github-pat Modal secret to enable push; changes are
committed locally on the volume"`). Users who only care about read/edit
get zero-setup v1; users who want push set the secret separately.

## Recommended mechanism: bare cache + worktrees + blob:none

Three-layer scheme on the Modal Volume:

1. **Shared bare cache** at `/vol/repo-cache/<host>/<org>/<repo>.git`.
   - Created on first sighting of a URL via
     `git clone --bare --filter=blob:none <url>`.
   - Shared across every thread touching that repo — cloned once, reused
     forever.
   - Partial clone (`--filter=blob:none`) skips file-content blobs at
     clone time; they're fetched on-demand when files are opened. Makes
     even large-repo cold clones small (~metadata only).

2. **Per-thread worktree** at `/vol/workspaces/<thread_id>/`
   via `git -C <bare> worktree add --force <workspace_path> <ref>`.
   - Shares `.git/objects` with the bare cache — disk cost is O(files
     actually checked out), not O(repo-size × thread-count).
   - Worktree creation materializes HEAD's tree, typically 1–3s.
   - Stable per `thread_id` → Claude Code's `~/.claude/projects/<hash>/`
     continuity preserved, `--continue` keeps working unchanged.

3. **Refresh policy**
   - **New thread**: ensure bare cache (clone if missing), `git fetch
     --filter=blob:none origin <ref>`, then `worktree add`.
   - **Resumed thread**: worktree already exists, skip fetch by default
     (preserves Claude's view of the codebase between turns). Opt-in
     refresh via `/refresh` slash command in v2.

### Expected cold-start budget

| Phase | Cold (first thread, repo never seen) | Warm (bare cache exists) | Resumed thread |
|---|---|---|---|
| Modal container | 1–5s | 1–5s | 1–5s |
| Bare clone (`blob:none`) | 2–8s | 0 | 0 |
| `git fetch` | 0 (fresh) | <1s | 0 |
| `git worktree add` | 1–3s | 1–3s | 0 (exists) |
| Claude Code startup | ~1s | ~1s | ~1s |
| **Total** | **~5–17s** | **~3–9s** | **~2–6s** |

Warm path is dominated by Modal container spin-up — the right place to be.

## Files to create / modify

### New: `apps/delulu_sandbox_modal/src/delulu_sandbox_modal/repo_provisioner.py`

Runs *inside* the Modal sandbox. Pure stdlib + `subprocess`. No structlog
dependency at module level (the sandbox import path is sensitive — see
`fix: install structlog in sandbox image` history).

```python
REPO_CACHE_ROOT = "/vol/repo-cache"
WORKSPACES_ROOT = "/vol/workspaces"

def provision_workspace(
    thread_id: int,
    repo_url: str | None,
    ref: str = "HEAD",
) -> str:
    """Return absolute workspace_path. Idempotent. Raises on unrecoverable failure."""

def _bare_cache_path(repo_url: str) -> str:
    """Parse repo_url -> /vol/repo-cache/<host>/<org>/<repo>.git"""

def _ensure_bare_cache(repo_url: str) -> str:
    """git clone --bare --filter=blob:none if missing. flock-protected."""

def _fetch_bare(bare_path: str, ref: str) -> None:
    """git -C <bare> fetch --filter=blob:none origin <ref>. flock-protected."""

def _ensure_worktree(bare_path: str, workspace_path: str, ref: str) -> None:
    """git -C <bare> worktree add --force <workspace_path> <ref>, or checkout
    if the worktree already exists. flock-protected per workspace."""

def _parse_repo_url(repo_url: str) -> tuple[str, str, str]:
    """Return (host, org, repo) for directory layout."""
```

**Locking**: `fcntl.flock(LOCK_EX)` on `.provision.lock` files inside the
bare cache dir and the workspace dir. Two concurrent threads on the same
repo will serialize cleanly. Verify flock semantics on Modal Volume as
part of testing — if not honored, fall back to `os.link`-based atomic
rename locks.

**Dead/edge cases** to handle:
- Workspace exists but isn't a worktree (legacy stub layout): `rm -rf`,
  recreate.
- Bare cache deleted out from under a worktree: recover by nuking the
  worktree and recreating.
- Running `git -C <bare> worktree prune` on each provision to clean up
  stale registrations cheaply.

### Modify: `apps/delulu_sandbox_modal/src/delulu_sandbox_modal/app.py`

Change `run_claude_code` signature:

```python
def run_claude_code(
    session_id: str,
    thread_id: int,
    prompt: str,
    *,
    repo_url: str | None = None,
    ref: str = "HEAD",
    resume: bool = False,
) -> str:
```

- Drop `workspace_path` from the signature — the sandbox owns provisioning
  now. Derive via `provision_workspace(thread_id, repo_url, ref)` after
  credential setup, before the `subprocess.run`.
- Log `provision.timing` with `cold_clone_ms`, `fetch_ms`, `worktree_ms`,
  `total_ms` — primary observability for the cold-start work.
- `volume.commit()` at the end already covers bare cache + worktree state.

### Modify: `apps/delulu_discord/src/delulu_discord/dispatcher.py`

`SandboxDispatcher.run_task` signature change:

```python
async def run_task(
    self,
    session_id: str,
    thread_id: int,
    prompt: str,
    *,
    repo_url: str | None = None,
    ref: str = "HEAD",
    resume: bool = False,
) -> str:
```

Pass everything through to `self._fn.remote(...)`. Drops `workspace_path`
since the sandbox derives it now.

### Modify: `apps/delulu_discord/src/delulu_discord/session_manager.py`

- `Session` dataclass gains `repo_url: str | None` and `ref: str` fields.
- `create_session(thread_id, repo_url=None, ref="HEAD")` actually stores
  the args on the Session.
- `workspace_path` becomes a derived property
  (`f"/vol/workspaces/{self.thread_id}"`) for logging; no longer
  authoritative — the sandbox `provision_workspace` is the source of truth.

### New: `apps/delulu_discord/src/delulu_discord/repo_config.py`

Thin wrapper around `modal.Dict.from_name("discord-orchestrator-repo-config",
create_if_missing=True)` for persistent channel → repo binding. Lives on
the bot side (bot has Modal client auth via `/root/.modal.toml`).

```python
class RepoConfig:
    def __init__(self) -> None:
        self._dict = modal.Dict.from_name(
            "discord-orchestrator-repo-config", create_if_missing=True
        )

    def get(self, channel_id: int) -> tuple[str, str] | None:
        """Return (repo_url, ref) for a channel, or None if unbound."""

    def set(self, channel_id: int, repo_url: str, ref: str = "HEAD") -> None: ...

    def unset(self, channel_id: int) -> None: ...
```

### Modify: `apps/delulu_discord/src/delulu_discord/main.py` + `handlers.py`

- Register slash commands via `discord.app_commands` (already supported in
  the `discord.py` version already in the lockfile). Two commands:
  - `/setrepo url:<str> ref:<str=HEAD>` — bind repo to channel
  - `/unsetrepo` — clear the channel binding
  - `/commit message:<str>` — commit and (try to) push via optional PAT
- `on_message` handler, when dispatching a new thread, looks up the
  channel's binding via `RepoConfig.get(channel.id)` and passes
  `repo_url` / `ref` into `dispatcher.run_task(...)`.
- Thread replies inherit the session's stored `repo_url` / `ref` from the
  `SessionManager` — no binding lookup on every reply.

### Modify: `apps/delulu_discord/src/delulu_discord/settings.py`

Add:
```python
repo_cache_root: str = "/vol/repo-cache"
default_git_ref: str = "HEAD"
provision_lock_timeout_seconds: int = 60
```

### Note: the legacy `workspace.py` stub is already gone

The earlier orphaned `workspace.py` was removed when the two apps were
split, so there's nothing to delete here — just grep for
`ensure_workspace` before wiring in the new provisioner to confirm no
stragglers crept back in.

## Commit-back flow (`/commit`)

- Sandbox side: `/commit` dispatches a separate Modal function
  (`commit_workspace`) — or a `mode=commit` path on `run_claude_code` —
  that runs inside the same workspace, checks out (or creates) branch
  `claude/<thread_id>`, runs `git add -A && git commit -m "<message>"`,
  then attempts `git push`.
- The push rewrite injects the PAT only at push time via
  `git -c http.extraheader="Authorization: Basic ..."`, never persisted
  into `.git/config`.
- If `GITHUB_TOKEN` env var is missing (no `github-pat` secret configured):
  - Still do the local commit so state is preserved on the volume.
  - Return a clear message: "*committed locally on the volume; configure
    `github-pat` Modal secret to enable push*".
- Document the setup in README: `modal secret create github-pat
  GITHUB_TOKEN=<pat>`.

## Concurrency

Three race scenarios, all covered by `flock`:

1. **Two threads, same repo, cold cache.** Both try to `git clone --bare`.
   Exclusive lock on
   `/vol/repo-cache/<host>/<org>/<repo>.git.lock` before the existence
   check. Loser blocks, then sees the cache and proceeds.
2. **Two threads, same repo, concurrent fetches.** Same lock held during
   fetch. Warm fetches are <1s so contention is fine.
3. **Same-thread concurrent dispatches.** Per-workspace flock at
   `/vol/workspaces/<thread_id>/.provision.lock`. (The bot should also
   serialize per-thread dispatches at the session level, but that's a
   pre-existing concern, not new.)

Acquire with 60s timeout; surface a clean error on timeout.

## Verification

1. **Flock semantics check.** One-off Modal function test that two
   concurrent `.spawn()` calls serialize on the same lock file. Must
   pass before shipping — load-bearing assumption of the concurrency
   design.

2. **Microbenchmark.** `tools/bench_provision.py` (not deployed) invoked
   via `modal run`, hits `provision_workspace` against three repo sizes
   (small <10MB, medium ~100MB, large >500MB) in all three path types
   (cold cache / warm new thread / resumed thread). Capture wall-clock
   per phase.

3. **End-to-end Discord smoke test**:
   - `/setrepo <url>` in a test channel → bind succeeds.
   - `@corchestra summarize the README` → first invocation creates bare
     cache and worktree, result posts in thread.
   - Follow-up reply (no @mention) → `claude --continue` resumes in the
     same workspace, confirms the cwd-stable session continuity
     invariant still holds.
   - `/commit "test commit"` without PAT → local commit succeeds,
     "configure github-pat" message shown.
   - Configure `github-pat`, re-run `/commit` → push succeeds.

4. **`provision.timing` logs** captured on Modal dashboard after a week
   of real use → check p50 cold vs warm to validate the budget table
   above.

## Critical files for implementation

All paths are relative to the repo root:

- `apps/delulu_sandbox_modal/src/delulu_sandbox_modal/app.py` — signature change + integrate `provision_workspace`
- `apps/delulu_sandbox_modal/src/delulu_sandbox_modal/repo_provisioner.py` — **new**, all git logic
- `apps/delulu_discord/src/delulu_discord/dispatcher.py` — pass `repo_url` / `ref` through to `run_claude_code`
- `apps/delulu_discord/src/delulu_discord/main.py` — register slash commands (`/setrepo`, `/unsetrepo`, `/commit`)
- `apps/delulu_discord/src/delulu_discord/handlers.py` — look up channel binding on dispatch
- `apps/delulu_discord/src/delulu_discord/repo_config.py` — **new**, `modal.Dict`-backed binding store
- `apps/delulu_discord/src/delulu_discord/session_manager.py` — store `repo_url` / `ref` on Session
- `apps/delulu_discord/src/delulu_discord/settings.py` — new config fields

## Out of scope — park for v2

- Private repo support (requires PAT-at-clone auth rewriting).
- `/refresh` slash command to re-fetch on a resumed thread.
- Workspace GC / TTL for volume size bounding.
- Auto-PR via `gh` CLI.
- Submodules, Git LFS.
- `min_containers=1` warm pool.
