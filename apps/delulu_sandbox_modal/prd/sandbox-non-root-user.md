# Run the Modal sandbox as a non-root user

Followup to PRs #53 and #54. Park for later — not urgent today.

This is a *plan*, not a spec of implemented behavior. Nothing in this
document is in the code yet.

## Context

The v1 repo-provisioning feature ended up needing Claude Code to
execute `Edit`/`Write`/`Bash` tool calls inside the sandbox, which
the default `-p` (non-interactive) permission mode refuses. Two
attempts to unblock it:

1. **PR #53** — added `--dangerously-skip-permissions`. Shipped to
   prod and immediately crashed at startup with:

       Claude Code exited with code 1:
       --dangerously-skip-permissions cannot be used with
       root/sudo privileges for security reasons

   Modal containers run as root by default. Claude Code
   unconditionally blocks bypass mode under root.

2. **PR #54** — replaced the bypass flag with an explicit
   `--allowedTools` whitelist (Bash, Edit, Write, Glob, Grep,
   NotebookEdit, Read, Task, TodoWrite, WebFetch, WebSearch). The
   per-tool allowlist is a targeted grant, not a blanket bypass, so
   Claude Code doesn't apply the root check. This is what's
   running in prod today and unblocked the Edit tool path.

The `--allowedTools` workaround is stable and working. But it's a
workaround — the "correct" fix is to stop running the sandbox as
root in the first place. This PRD documents what that would take
and what would trigger doing it.

## Why this isn't urgent

`--allowedTools` is working today. The failure modes worth
considering:

- **Claude Code deprecates `--allowedTools`**. Unlikely soon — the
  CLI reference lists it as a documented flag with pattern-matching
  syntax, so Anthropic is clearly invested in it. Not zero risk
  though.
- **Claude Code starts rejecting `--allowedTools` under root too**.
  Possible but not observed. The root check in
  `--dangerously-skip-permissions` exists because the bypass mode
  is a blanket grant; extending that check to explicit per-tool
  allowlists would be a meaningful policy shift that would break a
  lot of users, not just us.
- **We want a tool that isn't in our allowlist**. Fail-closed —
  the new tool would get refused, we'd see the ✗ in the status
  message, and we'd add it to the list. Cheap to fix.

So the current setup is "works fine until one of three specific
upstream changes happens." Good enough for single-user bot scope.
Not good enough for a production product with tight uptime
requirements on a dependency you don't control.

## Why we might want to do it anyway

Three forces push in the direction of non-root regardless of
Claude Code's policy:

1. **Defense in depth.** Running any network-facing subprocess as
   root is a worse default than running it as a constrained user,
   regardless of how well-sandboxed the container layer is. The
   Modal sandbox is already a strong boundary, but layered security
   is cheap insurance.

2. **Forward compatibility.** Claude Code is evolving quickly. If
   Anthropic tightens the permission model further (or adds more
   root checks, or introduces sudo-aware tools), we'd want to be
   on the right side of that change before it hits us. Being
   non-root makes future CC upgrades smaller bets.

3. **Matches the production Claude Code deployment model.**
   Anthropic's own docs for running Claude Code in CI/CD recommend
   non-root. We're currently the odd one out. Alignment with the
   recommended path reduces the chance of hitting edge cases that
   the Anthropic team never tests.

None of these are blocking today. They're the kind of thing you do
once the feature is stable, users are on it, and you have a
half-day of bandwidth to clean up the rough edges.

## Goal

At the end of this work, the Modal sandbox images
(`sandbox_image` used by `run_claude_code`, `provisioner_image`
used by `provision_workspace` and `commit_workspace`) run their
Python entry points as a dedicated non-root user — probably
`claude` or `sandbox` — with UID ≥ 1000. Claude Code invocations
can use `--dangerously-skip-permissions` (or
`--permission-mode bypassPermissions`) without hitting the root
check, which collapses the tool whitelist back to a single
general-purpose bypass flag and removes the fail-closed-on-new-
tools concern.

## Non-goals

- **Switching off the `--allowedTools` pattern before the non-root
  image work is fully validated.** The explicit whitelist is
  strictly more informative in failure modes (you see which tool
  got refused) and can coexist with non-root as a belt-and-
  suspenders measure. If we ship this work, we still might want to
  keep the whitelist for observability.
- **Changing the trust model**. The Modal sandbox stays the trust
  boundary. Non-root inside the sandbox is layered defense, not a
  replacement for container isolation.
- **Migrating the bot-side Docker image**. The Discord bot is a
  different process that runs on a VPS droplet, not inside Modal.
  It's already in its own Dockerfile and its own trust domain.
- **GID / capability isolation beyond the basic `useradd` default**.
  Fancier hardening (drop-capabilities, seccomp profiles, readonly
  rootfs) is out of scope — Modal's container runtime handles a
  lot of that for us already.

## The concrete work

### 1. Add a non-root user to `sandbox_image`

```python
sandbox_image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("git", "curl")
    .pip_install("structlog>=24.0")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @anthropic-ai/claude-code",
        # NEW: create a non-root user with a real home directory
        "useradd -m -u 1000 -s /bin/bash claude",
    )
    .add_local_python_source("delulu_sandbox_modal")
)
```

Same pattern for `provisioner_image` (which already has git +
ca-certificates, just needs the `useradd`).

### 2. Tell Modal to run the function as that user

Modal exposes a `user` parameter — or the equivalent via
`Dockerfile` `USER` directive — for specifying the runtime user.
The exact API name needs to be confirmed against the Modal version
we're on. Candidates:

- `@app.function(user="claude", ...)` — explicit kwarg if Modal
  supports one
- `modal.Image.dockerfile_commands("USER claude")` — falls through
  to Docker semantics if no explicit kwarg
- `modal.Image.run_commands` at the end of the image spec —
  sets the default user for subsequent containers

Need to verify which of these Modal actually supports in v1.x.
If none of them are first-class, we may need to `su - claude -c
<command>` wrap the entry point, which is ugly but works.

### 3. Fix `/vol` ownership

Modal Volumes are mounted into the container with specific
ownership. If the volume was created before the non-root user
existed, the existing files on it (`/vol/claude-home`,
`/vol/workspaces/*`, `/vol/repo-cache/*`) are owned by root.
The non-root user won't be able to write to them.

Options:

- **One-time `chown` at container start**, before the user
  switch. Runs once per container spin-up, takes milliseconds for
  the root-owned directories we care about. Simplest path.
- **Set volume ownership via Modal's volume config**, if Modal
  exposes that as a parameter. Cleaner but depends on Modal's API.
- **Recreate the volume**. Destructive — loses the bare cache and
  all existing thread workspaces. Forces every user to redo all
  prior work. Last resort.

Probably go with option 1: a small entry-point shim that
`chown`s `/vol/claude-home` and `/vol/workspaces` to the non-root
user, then `exec`s the actual function under that user. Cost is
sub-second at container start and idempotent (no-op after the
first run).

### 4. Fix `node_modules` ownership

Global npm install puts Claude Code at
`/usr/lib/node_modules/@anthropic-ai/claude-code/`, owned by
root. The `claude` binary at `/usr/bin/claude` is a symlink into
that directory. Non-root should still be able to execute it (the
dir has `o+rx` by default) — but if Claude Code tries to write
to its own install directory (e.g. for auto-update or cache),
it'll fail.

Mitigations:

- Rely on the `HOME=/vol/claude-home` env var already in use to
  route all CC state writes to the volume, not the install dir.
  This is already the case — Claude Code writes to
  `~/.claude/` which we point at the volume.
- Verify no `pip install claude-code` or `npm install` commands
  run at function runtime; they're all baked into the image.
- Keep `apt_install` and `pip_install` in the image build phase,
  where they run as root.

None of these are new — the existing setup already routes writes
to `/vol`. The non-root switch should be transparent if we verify
the assumption.

### 5. Validate `claude --continue` still works

The session resume mechanism keys off `HOME` + cwd. Our setup
has `HOME=/vol/claude-home` and cwd
`/vol/workspaces/<thread_id>`. Under the new user:

- `/vol/claude-home` needs to be writable by the non-root user
  (covered by step 3's chown)
- `/vol/workspaces/<thread_id>` same
- The `.credentials.json` file that seeds the OAuth token on first
  run needs to be readable by the non-root user AND writable so CC
  can rotate the refresh token

The credential-file permission bits are specifically `0o600` in
the current seed logic. After chown, this is fine for the non-root
user but won't be readable by root anymore. Smoke-test after the
switch to confirm the token rotation loop still works.

### 6. Validate `git worktree add` across user boundaries

`provision_workspace` creates a worktree with `git -C <bare>
worktree add`. If the bare cache was created by root (previous
provisions) and the worktree is being created by non-root, git
might complain about cross-user access.

Options:

- `chown -R <user> /vol/repo-cache` at container start (covered
  by step 3's chown, just needs to include repo-cache)
- Create the non-root user with matching UID/GID to whatever
  wrote the cache previously — doesn't work if the cache is mixed
- Recreate the bare caches — same as volume recreation, loses
  work

Simplest: include `/vol/repo-cache` in the chown at container
start. Cost is O(files in all bare caches), which is metadata-
only, so still millisecond range.

### 7. Collapse the `--allowedTools` whitelist (or don't)

Once non-root is stable, we CAN switch back to
`--dangerously-skip-permissions` or
`--permission-mode bypassPermissions`. Should we?

**Arguments for bypass mode:**

- Single flag, no maintenance as Claude Code adds tools
- Closer to the "do whatever you need to" semantics we actually
  want inside the sandbox
- Matches the documented recommendation for scripted / CI use

**Arguments for keeping the whitelist:**

- Fail-closed on unknown tools is a useful observability signal —
  if Claude starts trying to use a new tool we didn't expect, we
  want to know
- Explicit allowlists document intent better than "allow
  everything"
- Zero downside to keeping it under non-root (no root check to
  dodge)

I lean toward **keep the whitelist as the primary mechanism**
even after the non-root migration. The whitelist is cheap to
maintain (new tools in Claude Code are rare) and the failure mode
is strictly more informative. Only switch to bypass mode if
whitelist maintenance ever becomes annoying in practice, which I
don't expect it will.

## Risks

1. **Modal container user API doesn't exist or doesn't work as
   expected.** The whole plan depends on Modal letting us specify
   a non-root runtime user. If Modal 1.x doesn't support this
   directly, we'd need to wrap the entry point in a `su` shim,
   which is uglier but still works. Verify this first.

2. **`/vol` ownership migration strains.** The `chown -R` at
   container start is fine for small volumes but gets expensive
   as the bare cache grows (hundreds of MB per repo × N repos).
   Might need a "chown once, stamp a marker, skip on subsequent
   starts" optimization. Not needed for v1 but worth noting.

3. **Claude Code internally expects root for some feature we
   haven't noticed.** Unlikely but possible — CC is evolving and
   we're not on the documented-happy-path for every feature.
   Smoke-test every tool (Read, Edit, Write, Bash, Grep, Glob,
   Task, etc.) after the switch.

4. **Node.js / npm permissions break.** The `claude` CLI is a
   Node.js binary. Node's default module resolution might try to
   read/write paths that aren't accessible to the non-root user.
   Specifically: `~/.npm`, `/root/.config`, etc. We route `HOME`
   away from those, but there may be edge cases.

5. **Session-token refresh fails under non-root.** The
   `.credentials.json` file handles OAuth refresh in-place. If
   the non-root user can't write to the volume path (because of
   some subtle ownership issue after the chown), refreshes will
   fail silently and the bot will eventually hit auth errors
   when the token expires. Smoke-test by forcing a token
   rotation (set `.credentials.json`'s expiration manually, run
   Claude Code, verify the file got updated).

## Validation plan

After the image changes land but before shipping to prod:

1. **Build the new image locally** with `modal image build` (or
   the equivalent). Confirm the `useradd` succeeds and `/usr/bin/
   claude` is executable by the new user.

2. **Run a minimal `claude --help`** inside the image as the
   non-root user. Any startup error shows here.

3. **Spin up a throwaway Modal deploy** under a different app
   name (`discord-orchestrator-nonroot-test`) and run the existing
   smoke test from `docs/delulu-usage.md` end-to-end:
   - Cold clone
   - Warm cache
   - Read + Edit on README.md
   - `/commit` with a real PAT

4. **Compare `provision.timing` logs** against the root baseline.
   Non-root should be within 100ms of root for all ops — if
   there's a big regression, something's wrong with the chown
   or the volume mount.

5. **Confirm `--dangerously-skip-permissions` no longer crashes**
   under the new user. Can swap the flag in temporarily to verify,
   then swap it back to the whitelist per the "collapse" decision
   above.

6. **Smoke-test credential rotation.** Forcibly expire the
   OAuth token in `.credentials.json` (set `expires_at` to a past
   timestamp), run a Claude Code command, confirm the file on the
   volume got updated with a fresh token. If not, the refresh
   path is broken under non-root and needs more investigation.

7. **Roll out to the real `discord-orchestrator` app** via a
   normal CD push. Watch logs for ~24 hours.

## Trigger — when to actually do this work

Any of the following:

- Claude Code adds a new permission check that `--allowedTools`
  doesn't satisfy, and we need the fallback
- `--allowedTools` is deprecated in a Claude Code release
- A new CC tool we want to use isn't satisfied by the whitelist
  (e.g., a tool with `Bash(*)` pattern-matching semantics we
  can't express)
- A security audit flags "running Claude Code as root" as a P1
- We hit one of the risk scenarios above and need to migrate
  to recover
- Someone has a half-day of bandwidth and is tired of the
  workaround

None of these are blocking the current v1 feature set. Ship the
feature, run the bot, come back to this when the forcing function
shows up.

## Out of scope

- Running the Discord **bot** process as non-root (different
  container, different trust domain, not affected by this PRD)
- Hardening Modal's container runtime beyond what the base image
  provides (seccomp profiles, dropped caps, readonly rootfs) —
  Modal's runtime defaults handle this
- Migrating off Modal entirely to a different sandbox technology —
  that's a product decision, not a security cleanup
- Reducing the `--allowedTools` whitelist to a smaller subset —
  the list is intentionally broad to avoid fail-closed surprises;
  shrinking it is a separate concern
