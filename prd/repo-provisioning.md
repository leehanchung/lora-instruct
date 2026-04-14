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
provision_lock_timeout_seconds: int = 60
```

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
