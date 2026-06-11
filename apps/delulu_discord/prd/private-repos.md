# Private GitHub repo support for delulu

**Status:** v1-shipped · v2-parked

Extension of the repo-provisioning v1 plan
(`apps/delulu_discord/prd/repo-provisioning.md`) to cover private
GitHub repos. v1 explicitly deferred this — the clone path is
anonymous HTTPS today and the allowlist's `/admin_addrepo`
validation rejects anything `git ls-remote` can't see without
credentials.

This is a *plan*, not a spec of implemented behavior. Nothing in
this document is in the code yet.

## Context

The repo-provisioning v1 stack splits authentication along read
vs. write:

- **Clone / fetch (read).** `repo_provisioner.provision_workspace`
  in the sandbox runs `git clone --bare --filter=blob:none <url>`
  and `git fetch ... origin <ref>` against `https://github.com/...`
  with no credentials. Public repos work; private repos fail with
  GitHub's "Repository not found" error (which is the same response
  GitHub gives for "doesn't exist" — the API deliberately doesn't
  distinguish).
- **Push (write).** `/commit` already authenticates via the
  `github-pat` Modal secret, injected at push time via
  `git -c http.extraheader="Authorization: Basic ..."`, never
  persisted into `.git/config`. This is the existing "single user /
  single team" auth model — one shared PAT, one identity on
  `Author:` / `Committer:`, audit trail in Discord thread history
  rather than git blame.
- **Allowlist validation.** `/admin_addrepo` runs `git ls-remote
  https://github.com/<owner>/<repo>` on the bot side (not the
  sandbox) before adding to `RepoAllowlist`. No credentials. So
  even if the sandbox could clone a private repo, an admin can't
  add one to the allowlist in the first place.

All three need credentials to support private repos. None of them
need a *new* identity model — the v1 single-PAT design already
covers the "one team, one shared credential" case. The work is
extending the existing PAT to cover read paths and reworking
validation, not introducing per-user identity.

## Goals / non-goals

**Goals**
- Private GitHub repos clone/fetch via the existing single-team
  PAT model. Same `github-pat` Modal secret extended to cover read,
  or a second read-scoped secret — to be decided below.
- `/admin_addrepo` accepts private repos when the PAT can see them,
  rejects them with a clear "PAT can't see this repo" error
  otherwise (distinct from "repo doesn't exist").
- PAT material never lands in `.git/config`, on-disk URLs, or
  cached remotes. Same in-memory `http.extraheader` injection used
  by `/commit` today.
- Cold-start budget unchanged. Auth header injection is free; the
  network round-trips are identical to the public-repo path.

**Non-goals**
- Multi-user identity (per-user PATs, GitHub App with per-installation
  tokens). Still parked for v2 — see repo-provisioning.md "Out of
  scope" §"Multi-user identity via GitHub App." This PRD does NOT
  walk back the explicit warning against growing the shared-PAT
  scheme into a per-user dict.
- Non-GitHub git hosts (GitLab, Bitbucket, self-hosted). Same auth
  rewriting trick works in principle but each host has its own
  token format and each adds its own setup-doc surface area.
- SSH-based clone (`git@github.com:...`). HTTPS+PAT only. SSH would
  require deploying a key into the sandbox image and rotating it
  through Modal Secrets, more moving parts than the header
  injection path.
- Per-repo PAT scoping enforcement on the bot side. We document the
  recommended fine-grained PAT scope in the setup instructions but
  don't programmatically enforce that the configured PAT is
  scoped to exactly the allowlisted repos. (GitHub's fine-grained
  PAT API doesn't expose introspection of granted repos, so this
  isn't trivially checkable.)
- Token rotation automation. If the PAT expires, the operator
  rotates the Modal Secret manually. No refresh, no expiry
  warnings.

## User decisions locked in

| Question | Choice |
|---|---|
| One PAT or two? | **Reuse the existing `github-pat` Modal Secret** for both clone and push. One secret to rotate; matches v1's single-team scope. Operator-side PAT scope (fine-grained, allowlist-only) is the real least-privilege control, not a second secret. |
| Validation of private repos at `/admin_addrepo` | **Delegate to the sandbox** via a `validate_repo_access` Modal function. The secret is already mounted there for `/commit`; no need to add a second mount on the bot container. ~1s round-trip fits Discord's 3s interaction-token budget. |
| Behavior when PAT is missing at clone time | **Refuse-and-instruct,** scoped to repos tagged `private` on the allowlist. Public repos in the allowlist still work without the PAT — the message shape mirrors `/commit`'s no-PAT path. |
| Marker on allowlist entry | **Tag `public` / `private` at add time.** Dispatch-path check is a `RepoAllowlist` lookup instead of a network round-trip; visibility surfaces in `/admin_listrepos`. |
| `LiveStatus` subtitle when repo is private | **Prepend a 🔒** — `📁 🔒 alice/internal-service@main`. Cheap orientation cue for screen-share / audit. |
| Token rotation | **Manual** — `modal secret create github-pat GITHUB_TOKEN=<new> --force`. Operator-only, ~once a quarter; not worth slash-command surface. |
| Transport | **HTTPS+PAT only.** No SSH (would require baking a key into the sandbox image). |
| Identity model | **Stay inside v1's single-team auth scope.** Do NOT introduce per-user PATs in a Modal Dict, even as a stepping stone — that warning from `repo-provisioning.md` "Out of scope" §"Multi-user identity via GitHub App" still binds. |

### Note on access control via Discord channels

The per-channel `/setrepo` binding from v1 *is* the per-repo access
control surface for private repos. There is no separate ACL layer —
Discord channel permissions decide who can `@delulu` against which
repo:

- A private allowlisted repo is accessible to exactly the users who
  can post in channels bound to it (via `/setrepo`).
- Granting a user access to a private repo = inviting them to the
  channel where it's bound. Revoking = removing them from the
  channel (or `/unsetrepo`'ing the channel).
- The `MANAGE_GUILD`-gated allowlist is the *outer* ring (which
  repos can be bound at all in this server); per-channel binding +
  Discord channel membership is the *inner* ring (who can dispatch
  against which bound repo).

This works because the bot's whole identity model is single-team:
one shared PAT, one effective git identity, audit trail in Discord
thread history. If we ever scale to multi-team or per-user
identity (the parked GitHub App path), the channel-level access
model still works but needs revisiting alongside the auth rewrite.

## The auth rewriting mechanism

Same trick `/commit` uses today, applied to `git clone` and
`git fetch`:

```python
def _git_with_auth(args: list[str], pat: str | None) -> list[str]:
    """Prefix a git command with -c http.extraheader=... when PAT is set."""
    if pat:
        auth = base64.b64encode(f"x-access-token:{pat}".encode()).decode()
        return ["git", "-c", f"http.extraheader=Authorization: Basic {auth}", *args[1:]]
    return args
```

- `x-access-token` as the username is GitHub's documented
  convention for PAT-based HTTPS auth. The PAT goes in the
  password field. (The same convention used for GitHub App
  installation tokens.)
- The header is set via `-c` for the lifetime of one process
  invocation. It's never written to `.git/config`. Worktrees
  inherit nothing — every git invocation re-injects.
- **Verify the URL never contains the PAT.** Specifically do NOT
  rewrite the clone URL to `https://x-access-token:<PAT>@...` — that
  form persists into `.git/config`'s `[remote "origin"] url = ...`
  and we'd have to scrub it. Header injection avoids the problem.

## Files to create / modify

### Modify: `apps/delulu_sandbox_modal/src/delulu_sandbox_modal/repo_provisioner.py`

- Mount the `github-pat` Modal Secret on the `provision_workspace`
  function (same secret already mounted on `/commit`'s function).
- New private helper `_git_with_auth(args, pat)` as shown above.
- `_ensure_bare_cache(repo_url)` and `_fetch_bare(bare_path, ref)`
  use `_git_with_auth` when the PAT env var is set. Public-repo
  call sites are unchanged in behavior — no PAT, no header.
- New module-level `validate_repo_access(repo_url) -> "public" | "private" | "not_found"`
  Modal function for the bot's allowlist-add path. Runs
  `git ls-remote` once anonymously, once authenticated; classifies
  based on the response pair. Cheap (<2s end-to-end including
  Modal hop).
- No change to `provision_workspace`'s signature or its
  `max_containers=1` orchestration. Auth is a transparent
  pass-through.

### Modify: `apps/delulu_discord/src/delulu_discord/repo_allowlist.py`

- Allowlist entries grow a `visibility: "public" | "private"`
  field. Storage shape change: `list[str]` → `list[dict]` or
  `dict[str, str]` keyed by `owner/repo`. Latter is cleaner.
- **Sequencing:** land this schema change *after* the persistence fix
  in `setrepo-persistence-bug.md`. The allowlist store is the same
  `modal.Dict` whose `.aio` put-path may be silently dropping writes;
  shipping the `visibility` marker first would let it inherit that bug.
- Backward compatibility: on read, treat any bare `str` entry as
  `{"visibility": "public"}` (these are the v1 entries that
  predate the marker). New entries always carry the marker.
- New method `get_visibility(guild_id, owner_repo) -> str | None`
  for the dispatch path's "do we need PAT?" check.

### Modify: `apps/delulu_discord/src/delulu_discord/handlers.py`

- `/admin_addrepo` handler:
  1. Call `validate_repo_access.remote(repo_url)` instead of
     anonymous `git ls-remote`.
  2. On `"not_found"` — reject with "Repo doesn't exist or PAT
     can't see it. Check the URL and that the `github-pat` Modal
     Secret has access."
  3. On `"public"` or `"private"` — `RepoAllowlist.add(guild_id,
     owner_repo, visibility=...)`. Reply with the visibility
     reflected in the success message: "✅ Added
     `alice/api-service` (private) to the allowlist."

- `_dispatch_and_respond` (or wherever the per-dispatch repo lookup
  lives):
  - Look up channel binding via `RepoConfig.get(channel_id)`.
  - If bound and the allowlist entry is `private`, verify the PAT
    is reachable before dispatching. (Cheapest check: check
    `RepoConfig`-side metadata; we can mirror the visibility into
    `Session` at create time so dispatch doesn't re-hit the
    allowlist.)
  - If the PAT is missing — refuse-and-instruct, same message
    shape as `/commit`'s missing-PAT path.

### Modify: `apps/delulu_discord/src/delulu_discord/session_manager.py`

- `Session` dataclass gains `repo_visibility: str | None` (or just
  `is_private: bool`) so the dispatch path doesn't re-hit the
  allowlist on every reply. Set at session creation time from the
  allowlist lookup.

### Modify: `apps/delulu_discord/src/delulu_discord/streaming.py`

- `LiveStatus.__init__` takes an additional optional `is_private:
  bool` (defaults False).
- The repo-line render in `_render()` prepends `🔒 ` when
  `is_private` is True. Examples:

  ```
  💭 Thinking about your request...
  📁 🔒 alice/internal-service@main
  ```

  ```
  💭 Thinking about your request...
  📁 alice/api-service@main
  ```

### No change: `apps/delulu_sandbox_modal/src/delulu_sandbox_modal/app.py`

`run_claude_code` doesn't need to know about visibility — the
PAT mount is on `provision_workspace` and the auth-header
injection is internal to `repo_provisioner`. The only signature
churn would be optional and adds no behavior; skip it.

### Docs: `apps/delulu_discord/README.md` (or wherever PAT setup lives)

- Document that `github-pat` now needs **read** access to
  allowlisted private repos in addition to the existing **write**
  access for `/commit` push.
- Recommend a fine-grained PAT scoped to exactly the repos on the
  server allowlist, with permissions: `Contents: Read and write`,
  `Metadata: Read`. (Metadata is required by GitHub for any
  fine-grained PAT.)
- Document the rotation procedure: `modal secret create
  github-pat GITHUB_TOKEN=<new> --force` overwrites the existing
  secret; sandbox containers pick up the new value on next cold
  start (or immediately on next dispatch since each `.remote()`
  call hits a fresh container view of the secret).

## Threat model deltas

The v1 threat model in repo-provisioning.md "Access control and
threat model" still applies. The deltas this PRD introduces:

1. **PAT blast radius grows from "push only" to "push + read."**
   A leaked PAT now exposes private repo *contents*, not just the
   ability to push branches. Mitigated by the recommended
   fine-grained PAT scope (only allowlisted repos), but worth
   calling out — the operator's secret hygiene matters more under
   this PRD than under v1.
2. **Secret-echo risk is bigger.** The v1 PRD already noted that
   Claude can read config files and surface them in Discord
   messages even on public repos. With private repos in scope,
   an accidental message could leak proprietary code. The
   allowlist is still the primary control; nothing in this PRD
   weakens it. But the cost of a slip-up is higher — operators
   should think twice before allowlisting a repo with secrets in
   it (even read-protected secrets — a `.env` example file with
   real-looking placeholder values is still bad to leak).
3. **`/admin_addrepo` becomes PAT-gated.** Validating a private
   repo requires the PAT to be reachable from the sandbox at
   add-time. If the secret is missing, the admin gets a clear
   error and can't add private repos until they fix it. This is
   the correct failure mode — silently downgrading to
   anonymous validation would let admins think they added a
   private repo when they actually rejected it.
4. **Bare cache contents are now sensitive.** `/vol/repo-cache/`
   may contain private repo objects. The volume isn't shared
   outside the Modal app, but the operator should know that the
   "warm cache" optimization means cloned data sticks around on
   disk indefinitely. Workspace GC (still parked) becomes more
   relevant — see the v1 "Out of scope" section.

## Failure modes

| Scenario | Behavior |
|---|---|
| PAT missing when admin runs `/admin_addrepo` on a private repo | `validate_repo_access` returns `"not_found"` (anonymous and authenticated both fail). Admin sees the unified error. They check Modal Secrets, fix, retry. |
| PAT missing when user dispatches against a bound private repo | Refuse-and-instruct ephemeral reply on the @mention. Workspace not provisioned. Same message shape as `/commit`'s no-PAT path. |
| PAT expires between admin-add and user-dispatch | Same as "missing" from the dispatch path's perspective. Refuse-and-instruct. |
| PAT scope too narrow (allowlisted repo not in PAT's repo list) | `git clone` fails with auth error; `repo_provisioner` raises; dispatcher catches, posts a "PAT can't access this repo" error in thread. Admin updates the PAT scope in GitHub, runs `modal secret create --force`, retries. |
| PAT was created with `read` only and user runs `/commit` | Existing `/commit` flow's push fails with auth error; `/commit` already handles this. No new code path. |
| Allowlist entry was tagged `public` at add time but the repo flips to `private` upstream | Clone starts failing. Admin re-runs `/admin_addrepo` to refresh the marker (idempotent on the `owner/repo` key). |
| Allowlist entry was tagged `private` but the repo flips to `public` | Still works — authenticated clone of a public repo succeeds. We could add a `/admin_refreshrepo` for cleanliness but it's not necessary. |

## Verification

1. **Validation function unit-ish test.** A small `modal run`
   harness for `validate_repo_access` against three fixtures: a
   known-public repo, a known-private repo the PAT can see, a
   nonexistent repo. Confirm the three return values.
2. **End-to-end Discord smoke test additions** (extending the
   v1 verification list):
   - As admin, `/admin_addrepo repo:<your>/private-repo` →
     succeeds, success message reads `(private)`. `/admin_listrepos`
     shows the private marker.
   - User in the channel runs `/setrepo` → autocomplete shows the
     private repo. `@delulu summarize the README` → first
     invocation clones with auth, posts the summary. LiveStatus
     subtitle shows `🔒`.
   - Rotate the PAT (`modal secret create --force`). New
     dispatch on the same thread → uses the new PAT
     transparently. Old dispatches on resumed threads continue
     working from the existing worktree without re-fetch.
   - Wipe the PAT (`modal secret delete github-pat`). New
     dispatch on the private-bound channel → refuse-and-instruct.
     Public-bound channels in the same server keep working.
3. **Verify no PAT leakage on disk.** After a private clone,
   inspect `/vol/repo-cache/<host>/<org>/<repo>.git/config` and
   confirm `[remote "origin"] url = ...` is the bare HTTPS URL,
   no PAT embedded.

## Out of scope — park for v2 or later

These are NOT in scope for this PRD and stay aligned with v1's
parking lot:

- **Multi-user identity via GitHub App.** Still parked. The
  shared-PAT extension here does not preempt the App migration —
  it just covers the read path under the same single-team scope.
  The v2 migration (per-user installation tokens, OAuth callback,
  etc.) replaces both the read and write auth uniformly when it
  happens.
- **Per-user PATs in a Modal Dict.** Same warning as v1: do NOT
  build this. It's the worst of both worlds and makes the App
  migration harder.
- **SSH clone.** Out of scope — see Non-goals.
- **Non-GitHub git hosts.** Out of scope — see Non-goals.
- **Programmatic enforcement of PAT scope ⊆ allowlist.** GitHub's
  fine-grained PAT API doesn't expose granted-repo introspection
  cheaply. Document the recommendation, don't enforce.
- **Workspace GC for private-repo cache hygiene.** Still parked
  in v1. More relevant under this PRD (see threat model §4) but
  the design doesn't change — the same TTL/LRU policy works for
  public and private alike.
