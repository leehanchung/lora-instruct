# Efficient GitHub repo provisioning for delulu

## Context

The delulu bot is an @mention-gated interface to Claude Code
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
| Repo specification UX | Per-channel binding via `/setrepo` slash command. Argument is `repo:<owner>/<repo>` short form (not full URL), autocompleted from the server's allowlist. |
| Git auth | Public repos only for clone; single shared `github-pat` Modal secret for `/commit` push. Refuse-and-instruct when missing. |
| Warm container pool | No — cold container start is acceptable |
| Access control | Per-server allowlist stored in `modal.Dict`, managed via admin-only slash commands gated on Discord's `MANAGE_GUILD` permission. `/setrepo` rejects any repo not on the server's allowlist. See "Access control and threat model" below. |
| Identity / multi-user scope | Single-user or single-team model. All commits attribute to the PAT owner (i.e., you). Audit trail is the Discord thread history, not git blame. See "Scope: single user / single team" below. |
| Active-repo visibility | `LiveStatus` gains a second line under the thinking/reasoning header showing `📁 <owner>/<repo>@<ref>` when a repo is bound; rendered as a plain line (not a spoiler), omitted entirely when no repo is bound. |
| v1 scope | Provisioning + allowlist + `/commit` commit-back |

### Scope: single user / single team

v1 assumes the bot is operated by one person or one team working on a
shared set of repos. The `github-pat` Modal secret is a single shared
credential — every commit the bot makes is authenticated by that PAT,
and the git `Author:` / `Committer:` fields reflect the PAT owner's
identity. This is fine for a solo user or a small trusted team where
the Discord thread history is the real audit trail ("who asked for
what"), not git blame.

**What this explicitly does NOT handle:**

- Multiple independent users, each wanting commits attributed to their
  own GitHub identity. The right answer there is a **GitHub App** with
  per-user installations (OAuth callback, per-user installation
  tokens, token refresh), parked in v2 and called out in the "Out of
  scope" section. The single-PAT design does not scale to that; don't
  try to grow it sideways with a Modal Dict of per-user PATs — that's
  the worst of both worlds (security surface of stored PATs, friction
  of Discord-side onboarding, no actual identity preservation).

If v1 usage hits the limits of the single-team model, migrate the
auth layer to the GitHub App path rather than extending the shared-PAT
scheme.

### Note on the auth / commit-back tension

"Public repos only" means clone/fetch need no credentials. But `/commit`
requires *push*, which GitHub requires auth for even on public repos.
Resolution: clone path uses anonymous HTTPS; `/commit` path reads the
`github-pat` Modal secret and **refuses cleanly with setup instructions
if missing** (see "Commit-back flow" below for the exact message).
Users who only care about read/edit get zero-setup v1; users who want
push set the secret separately via `modal secret create`.

## Access control and threat model

An unlimited `/setrepo url:<anything>` is a denial-of-service waiting to
happen. The bot owner (the person running the Modal app and paying the
Claude Pro/Max subscription) needs to be able to say "these specific
repos are the only ones my bot works on in this Discord server."

### Threats the allowlist addresses

1. **Cost DoS.** Without a limiter, any channel member can `/setrepo
   url:torvalds/linux` and wedge the Modal volume with a 10+ minute
   clone that bills the bot owner for egress, disk, and compute.
2. **Accidental scope creep.** The bot stops being "Claude Code for
   my team's repos" and drifts into "general Claude Code for any
   GitHub repo," burning subscription quota on random third-party work.
3. **Secret-echo risk.** Even on public repos, Claude can read config
   files and surface them in Discord messages. Constraining the set of
   repos reduces the blast radius of an accidental leak.
4. **Unbounded disk pressure.** `/vol/repo-cache/` grows indefinitely
   without GC. An allowlist naturally bounds cache size to the number
   of registered repos × their sizes.

### Storage and admin model

The allowlist lives in a `modal.Dict` on the bot side, keyed by Discord
`guild_id`. Each value is a list of `owner/repo` short forms:

```
modal.Dict["discord-orchestrator-allowlist"]:
  {
    guild_id (int) -> list[str]  # ["alice/api-service", "alice-org/shared-lib"]
  }
```

Admin slash commands (gated on `MANAGE_GUILD` via
`@app_commands.default_permissions(manage_guild=True)`) manage the
allowlist per-server:

| Command | Effect |
|---|---|
| `/admin_addrepo repo:<owner>/<repo>` | Add a repo to this server's allowlist. Validates the URL via `git ls-remote` before accepting — rejects private or non-existent repos at bind time, not at first dispatch. |
| `/admin_removerepo repo:<owner>/<repo>` | Remove a repo from this server's allowlist. Autocompletes from the current list. Does NOT retroactively unbind channels that were already pointing at the repo; existing bindings stay intact until explicitly `/unsetrepo`'d. |
| `/admin_listrepos` | Show the current allowlist for this server. |

The `MANAGE_GUILD` gate means only users with the Discord
"Manage Server" permission (server owners, admins, and moderators who
have been granted that role) can edit the allowlist. Regular channel
members can `/setrepo` an allowlisted repo but cannot add new ones.

### `/setrepo` behavior under the allowlist

`/setrepo repo:<owner>/<repo> ref:<str=HEAD>`:

1. Parse `repo` as `owner/repo` short form. Reject with a clean error
   if the format is wrong.
2. Look up the server's allowlist via `RepoAllowlist.get(guild_id)`.
   If the repo isn't on the list, reply ephemerally with the list of
   allowed repos and tell the user how to request one be added
   (contact a `MANAGE_GUILD` role holder).
3. If accepted, derive the full URL (`https://github.com/<owner>/<repo>`)
   and store the binding via `RepoConfig.set(channel_id, repo_url, ref)`.

Discord's slash command autocomplete (`@app_commands.autocomplete` on
the `repo` argument) feeds from the server's allowlist, so users see a
dropdown of allowed repos instead of needing to know the exact names.

### Out-of-band setup

On first install in a new Discord server, the allowlist is empty and
the server admin must `/admin_addrepo` at least one repo before
`/setrepo` becomes useful. The bot's first-run DM to the installer
should mention this — "install complete. Add a repo with
`/admin_addrepo` (Manage Server permission required)."

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

See the revised budget table in the "Concurrency" section below,
which includes the `max_containers=1` orchestration hop. Summary:

- **Cold** (first thread on this repo, ever): ~6–19s
- **Warm** (bare cache exists): ~4–11s
- **Resumed thread**: ~2–6s (skips the provisioning hop entirely
  via a short-circuit in `run_claude_code`)

Warm and resumed paths are still dominated by Modal container
spin-up — the right place to be.

## Files to create / modify

### New: `apps/delulu_sandbox_modal/src/delulu_sandbox_modal/repo_provisioner.py`

Runs *inside* the Modal sandbox. Pure stdlib + `subprocess`. No structlog
dependency at module level (the sandbox import path is sensitive — see
`fix: install structlog in sandbox image` history).

This module exposes a **Modal function**, not a plain Python function.
The `@app.function(max_containers=1)` decorator is load-bearing — it's
how we get cross-container serialization of provisioning work after
the flock scout (see "Concurrency" below) proved that filesystem-based
locks are a dead end on Modal Volumes.

```python
REPO_CACHE_ROOT = "/vol/repo-cache"
WORKSPACES_ROOT = "/vol/workspaces"

# Dedicated provisioning function. max_containers=1 means Modal
# serializes all concurrent invocations to a single container,
# giving us a global mutex on git operations without writing any
# lock code of our own.
@app.function(
    image=provisioner_image,
    volumes={"/vol": volume},
    max_containers=1,
    timeout=300,
)
def provision_workspace(
    thread_id: int,
    repo_url: str | None,
    ref: str = "HEAD",
) -> str:
    """Return absolute workspace_path. Idempotent. Raises on unrecoverable failure.

    Called from `run_claude_code` via `provision_workspace.remote(...)`.
    The `.remote()` hop routes through Modal's orchestration layer,
    which queues the call behind any other in-flight invocation.
    """

def _bare_cache_path(repo_url: str) -> str:
    """Parse repo_url -> /vol/repo-cache/<host>/<org>/<repo>.git"""

def _ensure_bare_cache(repo_url: str) -> str:
    """git clone --bare --filter=blob:none if missing. No locks needed
    because we're inside the serialized provision_workspace container."""

def _fetch_bare(bare_path: str, ref: str) -> None:
    """git -C <bare> fetch --filter=blob:none origin <ref>. No locks needed
    (see above)."""

def _ensure_worktree(bare_path: str, workspace_path: str, ref: str) -> None:
    """git -C <bare> worktree add --force <workspace_path> <ref>, or checkout
    if the worktree already exists. No locks needed (see above)."""

def _parse_repo_url(repo_url: str) -> tuple[str, str, str]:
    """Return (host, org, repo) for directory layout."""
```

**No locking code.** Because `provision_workspace` is a Modal function
with `max_containers=1`, only one invocation runs at any moment across
the entire Modal app. All git operations inside it are effectively
single-threaded — there is no second worker to race against. The
earlier design's `fcntl.flock` on `.provision.lock` files is gone
entirely. The flock scout at
`apps/delulu_sandbox_modal/tools/verify_flock.py` documents why
(filesystem locks don't propagate across Modal containers — neither
`fcntl.flock`, `os.mkdir` atomic EEXIST, nor `os.link` — the last of
which Modal Volumes don't even support).

**Commit + reload dance.** After `provision_workspace` finishes its
git work, it calls `volume.commit()` before returning. The caller
(`run_claude_code`, in its own container) then calls
`volume.reload()` to pick up the committed provisioning state before
cd-ing into the workspace. Without the reload, the calling container
still has its pre-provision view of the volume mounted and won't see
the worktree.

**Dead/edge cases** to handle:
- Workspace exists but isn't a worktree (legacy stub layout): `rm -rf`,
  recreate.
- Bare cache deleted out from under a worktree: recover by nuking the
  worktree and recreating.
- Running `git -C <bare> worktree prune` on each provision to clean up
  stale registrations cheaply.
- Resumed-thread short-circuit: if the workspace already exists AND
  a marker file records the same (repo_url, ref) as the current call,
  skip the git ops and return the path immediately. Shaves ~1–2s off
  the warm path.

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

- Drop `workspace_path` from the signature — the sandbox owns
  provisioning now. Derive via
  `workspace_path = provision_workspace.remote(thread_id, repo_url, ref)`
  after credential setup, before the `subprocess.run`.
- **`.remote()`, not a plain call.** `provision_workspace` is a
  separate Modal function with `max_containers=1`; calling it via
  `.remote()` routes through Modal's orchestration layer, which
  queues behind any other in-flight provisioning. A plain Python
  call would run inline in the `run_claude_code` container and
  bypass the serialization — guaranteeing races.
- **`volume.reload()` after the remote call.** Without this the
  calling container still has its pre-provisioning view of the
  volume mounted and won't see the newly-created worktree. The
  reload is cheap (<0.5s) and picks up the committed state.
- **Short-circuit for resumed threads.** If `resume=True` AND the
  workspace directory already exists on the mounted volume, skip
  the `provision_workspace.remote()` hop entirely — the existing
  worktree is what we want. Shaves ~1–2s off every resumed turn.
- Log `provision.timing` with `cold_clone_ms`, `fetch_ms`,
  `worktree_ms`, `total_ms` — primary observability for the
  cold-start work.
- `run_claude_code` itself keeps the default unlimited concurrency
  — we only serialize the git operations, not the Claude Code
  execution that takes minutes per call.

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

### New: `apps/delulu_discord/src/delulu_discord/repo_allowlist.py`

Companion to `repo_config.py` — a thin wrapper around
`modal.Dict.from_name("discord-orchestrator-allowlist",
create_if_missing=True)` for per-server repo allowlists. Keyed by
Discord `guild_id`, values are lists of `owner/repo` short forms.

```python
class RepoAllowlist:
    def __init__(self) -> None:
        self._dict = modal.Dict.from_name(
            "discord-orchestrator-allowlist", create_if_missing=True
        )

    def get(self, guild_id: int) -> list[str]:
        """Return the list of allowed `owner/repo` entries for a guild.
        Returns [] if the guild has no allowlist yet."""

    def add(self, guild_id: int, owner_repo: str) -> None:
        """Add an `owner/repo` entry to a guild's allowlist. Idempotent."""

    def remove(self, guild_id: int, owner_repo: str) -> None:
        """Remove an `owner/repo` entry. No-op if not present."""

    def contains(self, guild_id: int, owner_repo: str) -> bool:
        """True iff the entry is on the guild's allowlist."""
```

Validation at add time happens in the `/admin_addrepo` command handler,
not in this module — the module is a pure store. The command handler
runs `git ls-remote https://github.com/<owner>/<repo>` before calling
`add()` to reject private/nonexistent repos immediately.

### Modify: `apps/delulu_discord/src/delulu_discord/main.py` + `handlers.py`

Register slash commands via `discord.app_commands`. Five commands total —
three user-facing, three admin. (Yes, `/setrepo` overlaps both lists; it's
user-facing but gated by the allowlist populated by admins.)

**User-facing (no permission gate):**

- `/setrepo repo:<owner>/<repo> ref:<str=HEAD>` — bind repo to channel.
  Rejects with the current allowlist shown if `repo` isn't on it.
  `repo` argument autocompletes from the server's allowlist via
  `@app_commands.autocomplete` fed by `RepoAllowlist.get(guild_id)`.
- `/unsetrepo` — clear the channel binding.
- `/commit message:<str>` — commit and (try to) push via optional PAT.

**Admin-only (gated on `MANAGE_GUILD` via
`@app_commands.default_permissions(manage_guild=True)`):**

- `/admin_addrepo repo:<owner>/<repo>` — validate via `git ls-remote`
  and add to the server's allowlist. Rejects private/nonexistent repos
  at bind time.
- `/admin_removerepo repo:<owner>/<repo>` — remove from allowlist.
  `repo` argument autocompletes from the current allowlist. Does NOT
  retroactively unbind channels already pointing at the repo — existing
  bindings persist until explicitly `/unsetrepo`'d.
- `/admin_listrepos` — show the server's current allowlist. Ephemeral
  response (only the admin sees it).

**Handler-side logic:**

- `on_message` handler, when dispatching a new thread, looks up the
  channel's binding via `RepoConfig.get(channel.id)` and passes
  `repo_url` / `ref` into `dispatcher.run_task(...)`. If no binding,
  dispatch against an empty workspace as today (general Q&A mode).
- Thread replies inherit the session's stored `repo_url` / `ref` from
  the `SessionManager` — no binding lookup on every reply, no re-check
  against the allowlist (bindings are grandfathered at thread creation).

### Modify: `apps/delulu_discord/src/delulu_discord/settings.py`

Add:
```python
repo_cache_root: str = "/vol/repo-cache"
default_git_ref: str = "HEAD"
```

(No `provision_lock_timeout_seconds` — the concurrency redesign
removed filesystem locks entirely. Serialization now lives in
Modal's orchestration layer via `max_containers=1`, which has its
own function-level timeout.)

### Modify: `apps/delulu_discord/src/delulu_discord/streaming.py`

`LiveStatus` already renders a thinking/reasoning header as the first
line of the status message and doesn't collapse the transcript at
`finalize_done` (already verified in `streaming.py:58-60`). Add an
**active-repo subtitle** that renders as the second line whenever a
repo is bound and is omitted entirely when there's no binding.

Shape with a repo bound:

```
💭 Thinking about your request...
📁 alice/api-service@main
```

```
||🧠 Reasoning: should probably read the middleware file first…||
📁 alice/api-service@main
🔧 Read app/middleware.py ✓
🔧 Grep "rate_limit" ✓
✅ Done • 14 tools • 42s
```

Shape with no repo bound (identical to today):

```
💭 Thinking about your request...
```

Concrete change:

- `LiveStatus.__init__` takes an optional `repo_url: str | None` and
  `ref: str` (both threaded through from `handlers._dispatch_and_respond`
  via the `Session`).
- `_render()` gains a second positional: a `repo_line: str | None`. When
  non-None, it's emitted between the header and the tool lines. When
  None, it's a no-op — the existing output is unchanged.
- `_render_header()` logic is unchanged; thinking stays first.
- `_truncate_to_limit()` treats the repo line as protected (never
  truncated — it's cheap, it's orientation, and dropping it makes the
  long-transcript overflow case harder to read).

This is 15–20 lines of new code plus a `LiveStatus.__init__` signature
change. The existing `streaming.py` test file should gain coverage
for the repo-bound and no-repo branches.

### Note: the legacy `workspace.py` stub is already gone

The earlier orphaned `workspace.py` was removed when the two apps were
split, so there's nothing to delete here — just grep for
`ensure_workspace` before wiring in the new provisioner to confirm no
stragglers crept back in.

## Commit-back flow (`/commit`)

- **Pre-flight check.** Before touching the workspace, verify the
  `github-pat` Modal secret is available (Modal injects it as
  `GITHUB_TOKEN`). If missing, **refuse-and-instruct**: reply
  ephemerally with
  > ❌ Can't commit — `github-pat` Modal secret missing or expired.
  > Run `modal secret create github-pat GITHUB_TOKEN=<pat>` on your
  > laptop, then re-run `/commit`. Your workspace changes are still
  > there.

  No local commit is made in this case — the working tree is left
  exactly as Claude left it. User re-runs after configuring the
  secret and the command proceeds end-to-end. This is simpler than
  commit-locally-and-instruct (no hidden state on the volume, no
  confusion about what "committed locally" means) and the no-PAT
  path is rare in practice — it's a first-time setup or
  token-expiration event, not a normal-use concern.

- **Happy path.** Dispatch a separate Modal function
  (`commit_workspace`) — or a `mode=commit` path on `run_claude_code`
  — that runs inside the same workspace, checks out (or creates)
  branch `claude/<thread_id>`, runs
  `git add -A && git commit -m "<message>"`, then pushes.
- The push rewrite injects the PAT only at push time via
  `git -c http.extraheader="Authorization: Basic ..."`, never persisted
  into `.git/config`.
- Document the setup in README: `modal secret create github-pat
  GITHUB_TOKEN=<pat>`.

## Concurrency

### What we tried first (and why it didn't work)

The original design serialized provisioning via `fcntl.flock(LOCK_EX)`
on lock files inside the Modal Volume, with `os.link`-based atomic
rename locks as a fallback. **Neither works on Modal Volumes**, and we
have empirical evidence:
`apps/delulu_sandbox_modal/tools/verify_flock.py` is a one-off scout
that spawns three concurrent workers and tests three filesystem lock
primitives in sequence. Results on a real Modal Volume:

| Primitive | Result | Why |
|---|---|---|
| `fcntl.flock(LOCK_EX)` | ❌ **FAIL** | Kernel-level locks don't propagate across Modal containers. Each container has its own kernel and its own inode cache; locking `/vol/foo.lock` in container A doesn't block container B's flock on the same path. |
| `os.mkdir` atomic EEXIST | ❌ **FAIL** | All three workers successfully created the same directory. The Modal Volume backend doesn't serialize metadata writes across containers — mkdir atomicity is a local-kernel invariant, not a distributed one. |
| `os.link` atomic rename lock | ⛔ **UNSUPPORTED** | `PermissionError: [Errno 1] Operation not permitted`. The volume filesystem doesn't support hard links at all. The PRD's original fallback path is unusable regardless of atomicity. |

Root cause: **Modal Volumes aren't a live shared kernel filesystem.**
They're commit/reload-synced object storage. Writes from one container
are invisible to another until explicit `volume.commit()` +
`volume.reload()` — which is fundamentally the wrong model for any
"spin on a lock file" primitive. The PRD's earlier concurrency design
assumed POSIX semantics that the volume backend doesn't provide.

Re-run the scout at any time to re-verify:
```
modal run apps/delulu_sandbox_modal/tools/verify_flock.py::verify
```

### What we do instead: `max_containers=1` on `provision_workspace`

Serialization lives at **Modal's orchestration layer**, not at the
filesystem. `provision_workspace` is defined as a Modal function with
`@app.function(max_containers=1)`. Modal guarantees that at most one
container running this function exists across the entire app at any
time. Concurrent `.remote()` calls from multiple `run_claude_code`
containers get queued by Modal and processed one at a time.

```python
@app.function(
    image=provisioner_image,
    volumes={"/vol": volume},
    max_containers=1,  # <- the entire concurrency design
    timeout=300,
)
def provision_workspace(thread_id: int, repo_url: str | None, ref: str = "HEAD") -> str:
    ...
```

Inside the body there are **no locks**. No `.provision.lock`, no
`flock`, no retry loops, no timeout math. It's just sequential git
operations on a freshly-reloaded volume view, followed by
`volume.commit()` at the end. The earlier designs' ~40 lines of lock
management code are gone.

### How the race scenarios resolve under the new design

1. **Two threads, same repo, cold cache.** Both trigger
   `run_claude_code`, both call `provision_workspace.remote(...)`.
   Modal queues the second call behind the first. The first container
   does the cold clone, commits the volume, returns. The second
   container mounts the volume fresh (so it sees the committed bare
   cache), skips the clone, runs the fetch, returns. No race, no lock
   code.
2. **Two threads, same repo, concurrent fetches.** Same shape.
   Sequential by construction.
3. **Same-thread concurrent dispatches.** The bot serializes per-thread
   dispatches at the session level (pre-existing, not new). Even if
   it didn't, Modal's queueing would handle it — the two calls would
   serialize and the second would short-circuit on the resumed-thread
   path.

### Tradeoff worth knowing

`max_containers=1` is a **global** mutex on provisioning, not
per-repo. If two people in different Discord channels trigger
provisioning on two different repos at the same time, their
provisioning requests queue up instead of running in parallel.

For the v1 "single user / single team" scope this is invisible — the
bot sees provisioning requests arriving seconds or minutes apart, not
concurrently. For a larger deployment it could add noticeable latency.
The migration path is to replace `max_containers=1` with a per-repo
coordination primitive (candidate: `modal.Dict.put(key, value,
skip_if_exists=True)` as a CAS-based lock registry, keyed by repo
URL), which would let different repos provision concurrently while
still serializing same-repo contention. That's explicit v2 work and
listed in "Out of scope — park for v2."

### Revised cold-start budget

The `max_containers=1` design adds a Modal-orchestration hop
(`run_claude_code` → `provision_workspace.remote()` → back) and a
`volume.reload()` to the critical path. Updated numbers:

| Phase | Cold (first thread, repo never seen) | Warm (bare cache exists) | Resumed thread |
|---|---|---|---|
| `run_claude_code` container start | 1–5s | 1–5s | 1–5s |
| `provision_workspace.remote()` hop | 1–2s | 1–2s | skipped |
| Bare clone (`blob:none`) | 2–8s | 0 | 0 |
| `git fetch` | 0 (fresh) | <1s | 0 |
| `git worktree add` | 1–3s | 1–3s | 0 (exists) |
| `volume.reload()` | <0.5s | <0.5s | 0 |
| Claude Code startup | ~1s | ~1s | ~1s |
| **Total** | **~6–19s** | **~4–11s** | **~2–6s** |

Slightly worse than the earlier budget (which assumed inline flock-
protected provisioning). The extra ~1–2s hop for warm and cold paths
is acceptable; resumed threads skip the hop entirely via the
short-circuit in `run_claude_code` (see that file's modification
note), so the most common path stays fast.

## Verification

1. **Flock scout — already run, PRD cites the result.** See
   `apps/delulu_sandbox_modal/tools/verify_flock.py`. Re-run any time
   to re-verify that filesystem locks still don't work on Modal
   Volumes. No need to re-run before shipping — the scout is
   documentation, not a gating check.

2. **Microbenchmark.** `tools/bench_provision.py` (not deployed)
   invoked via `modal run`, hits `provision_workspace.remote()`
   against three repo sizes (small <10MB, medium ~100MB, large
   >500MB) in all three path types (cold cache / warm new thread /
   resumed thread). Capture wall-clock per phase. Must run on a
   separate test volume (not `claude-workspaces`) to avoid polluting
   the production cache.

3. **End-to-end Discord smoke test**:
   - As a non-admin user, `/setrepo repo:alice/api-service` with an
     empty server allowlist → rejected with a "not in the allowlist"
     message.
   - As a server admin, `/admin_addrepo repo:alice/api-service` →
     validates via `git ls-remote`, succeeds. `/admin_listrepos` →
     shows the repo.
   - Non-admin user, `/setrepo repo:alice/api-service` → now succeeds;
     autocomplete dropdown on the `repo:` argument shows
     `alice/api-service`.
   - `@corchestra summarize the README` → first invocation creates bare
     cache and worktree, result posts in thread.
   - Follow-up reply (no @mention) → `claude --continue` resumes in the
     same workspace, confirms the cwd-stable session continuity
     invariant still holds.
   - `/commit "test commit"` without PAT configured → refused with the
     "configure github-pat" message; workspace unchanged (no hidden
     local commit on the volume).
   - Configure `github-pat`, re-run `/commit` → commit lands, push
     succeeds, branch `claude/<thread_id>` appears on the remote.
   - As a non-admin user, `/admin_removerepo` → rejected by Discord
     itself (MANAGE_GUILD permission missing), before the command
     handler runs.

4. **`provision.timing` logs** captured on Modal dashboard after a week
   of real use → check p50 cold vs warm to validate the budget table
   above.

## Critical files for implementation

All paths are relative to the repo root:

- `apps/delulu_sandbox_modal/src/delulu_sandbox_modal/app.py` — signature change + integrate `provision_workspace`
- `apps/delulu_sandbox_modal/src/delulu_sandbox_modal/repo_provisioner.py` — **new**, all git logic
- `apps/delulu_discord/src/delulu_discord/dispatcher.py` — pass `repo_url` / `ref` through to `run_claude_code`
- `apps/delulu_discord/src/delulu_discord/main.py` — register user + admin slash commands
- `apps/delulu_discord/src/delulu_discord/handlers.py` — look up channel binding on dispatch
- `apps/delulu_discord/src/delulu_discord/repo_config.py` — **new**, `modal.Dict`-backed channel→repo binding store
- `apps/delulu_discord/src/delulu_discord/repo_allowlist.py` — **new**, `modal.Dict`-backed per-server allowlist store
- `apps/delulu_discord/src/delulu_discord/session_manager.py` — store `repo_url` / `ref` on Session
- `apps/delulu_discord/src/delulu_discord/settings.py` — new config fields

## Out of scope — park for v2

- Private repo support (requires PAT-at-clone auth rewriting).
- `/refresh` slash command to re-fetch on a resumed thread.
- Workspace GC / TTL for volume size bounding.
- Auto-PR via `gh` CLI.
- Submodules, Git LFS.
- `min_containers=1` warm pool.
- Per-user dispatch rate limits (independent of allowlist — someone
  in an allowlisted channel can still spam `@corchestra` and burn
  subscription quota). Revisit if it becomes a real problem.
- **Per-repo provisioning coordination via `modal.Dict`.** The v1
  concurrency design uses `@app.function(max_containers=1)` on
  `provision_workspace` as a global mutex — every provisioning call
  serializes through one Modal container. That's the right tradeoff
  for single-team scope (the contention window is small, and the
  simplicity pays for itself) but doesn't scale gracefully if the
  bot ever serves a larger population where multiple users
  concurrently provision different repos. The v2 migration: remove
  `max_containers=1`, add a `modal.Dict`-backed lock registry keyed
  by repo URL, acquire the per-repo key via an atomic CAS primitive
  (candidate API: `modal.Dict.put(key, value, skip_if_exists=True)`
  — needs verification before committing), and surround the git
  operations with the per-repo acquire/release. Different repos then
  provision in parallel; same-repo contention still serializes.
  Write a second scout (`verify_modal_dict.py`) to empirically
  confirm Modal Dict CAS semantics work across containers before
  committing to this design.
- **Multi-user identity via GitHub App.** The v1 single-PAT model
  works for a solo user or a trusted team. Scaling to N independent
  users — each wanting commits attributed to their own GitHub
  identity, each with their own per-repo access grants — requires
  registering a GitHub App, adding an OAuth callback endpoint (Modal
  supports this via `@modal.asgi_app` / `@modal.web_endpoint`), a
  `/connect` slash command that DMs the user an install link, and
  per-user installation-token storage with refresh logic (~200 lines
  of new infra). Do NOT try to grow the shared-PAT scheme sideways
  into a multi-user model by storing per-user PATs in a `modal.Dict`
  — that's the worst of both worlds and makes migration to the App
  path harder later.
