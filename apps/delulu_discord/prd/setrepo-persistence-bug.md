# `/setrepo` binding lost after bot restart

Known regression in the repo-provisioning v1 feature set, surfaced
by the post-ship smoke test of Phases 1–4. Writeup here so the bug
is tracked until we have bandwidth to chase it properly — fix is
almost certainly one line, but I want to verify the hypothesis
empirically rather than guess.

This is a *plan*, not a fix. Nothing in this document is in the
code yet.

## Symptom

1. User runs `/setrepo repo:leehanchung/SMILE-factory` in a channel.
2. Bot replies ephemerally with `✅ Channel bound to
   leehanchung/SMILE-factory@HEAD. New @claude mentions in this
   channel will run against this repo.`
3. `@delulu whats in this repo` works correctly — the LiveStatus
   shows the `📁 leehanchung/SMILE-factory@HEAD` subtitle and the
   sandbox provisions a real worktree.
4. Bot container restarts (CD push, manual `docker restart`, or
   crash recovery).
5. User runs `@delulu whats in this repo` again. The bot falls
   through to the no-repo fast path — empty workspace, no subtitle
   line, Claude reports "working directory is empty." The binding
   is gone.
6. `/setrepo` re-run succeeds and re-binds. So the storage layer
   is writable; something about the persistence is wrong.

## Why this shouldn't be happening

`RepoConfig` is backed by `modal.Dict.from_name(
"discord-orchestrator-repo-config", create_if_missing=True)`. Modal
Dicts are durable — writes persist on Modal's side across bot
restarts by design. A bot process holding a stale handle to a Modal
Dict that was later wiped is the only sane way this would "forget,"
and we never call `.clear()` anywhere.

So either:

1. **Writes aren't actually persisting.** `/setrepo` thinks it
   succeeded (user sees the ✅) but the underlying `modal.Dict`
   write is silently no-op-ing.
2. **Reads are looking at the wrong dict.** The bot reconnects to
   a different backing store after restart — different environment,
   different dict name, different client auth.
3. **Writes and reads are both fine but something else is clearing
   the dict.** Unlikely given no code path does this, but worth
   ruling out.

## Suspected root cause

**Option 1 (writes silently failing) is the most likely.** PR #51
converted `RepoConfig.set` from the blocking dict-style shorthand
to an async `.aio()` call:

```python
# Before (pre-PR #51)
self._dict[channel_id] = {"repo_url": repo_url, "ref": ref}

# After (PR #51)
await self._dict.put.aio(channel_id, {"repo_url": repo_url, "ref": ref})
```

Modal's own `AsyncUsageWarning` on the pre-#51 code suggested
`await ...__setitem__.aio(...)` as the rewrite. I guessed `put.aio`
as a cleaner equivalent. If `put.aio(key, value)` has different
semantics from `__setitem__.aio(key, value)` — or doesn't exist at
all on `modal.Dict` — the write silently drops.

**Evidence that would confirm this:**

- `modal dict items discord-orchestrator-repo-config` returns an
  empty dict (or is missing entries for the channels that were
  `/setrepo`'d)
- Bot logs show `repo_config.set` firing (from the `structlog.info`
  call) but no corresponding Modal-side error

**Evidence that would rule this out:**

- The dict dump shows the expected entries, which means writes
  DID persist and reads are the broken path — pointing at Option 2
  or 3

`RepoAllowlist.add` and `.remove` use the same `put.aio` pattern, so
if this theory is right, **admin commands are also broken in the
same way**. Users haven't complained yet because `/admin_addrepo`
gets verified via a fresh `/admin_listrepos` in the same session,
where the in-process cache papers over the missing persistence.
After a bot restart the admin-added repos should also vanish.

## Diagnostic plan

Before changing any code, run this and capture the output:

```bash
# Dump the repo-config and allowlist dicts
modal dict items discord-orchestrator-repo-config
modal dict items discord-orchestrator-allowlist

# Check bot logs for the write path
docker logs disco 2>&1 | grep -E "repo_(config|allowlist)\." | tail -20

# Check for any Modal errors the bot swallowed
docker logs disco 2>&1 | grep -iE "(error|traceback|warning)" | tail -40
```

Then run `/setrepo repo:leehanchung/SMILE-factory` one more time
and immediately re-run `modal dict items discord-orchestrator-repo-config`.

- If the dict is still empty → writes are silently failing.
  Proceed to the fix in "Candidate fixes" below.
- If the dict has the expected entry → writes work. The bug is on
  the read path or the key type. Escalate to a deeper read-path
  investigation before applying any fix.

## Candidate fixes

### Fix 1 — use `__setitem__.aio` / `__delitem__.aio` literally

Match Modal's own warning suggestion word-for-word rather than
assuming the method name:

```python
# repo_config.py
async def set(self, channel_id: int, repo_url: str, ref: str = "HEAD") -> None:
    await self._dict.__setitem__.aio(channel_id, {"repo_url": repo_url, "ref": ref})

async def unset(self, channel_id: int) -> None:
    try:
        await self._dict.__delitem__.aio(channel_id)
    except KeyError:
        pass
```

Same for `repo_allowlist.py`'s `add` and `remove`.

This is ugly (calling dunder methods explicitly on the `.aio`
accessor is not idiomatic), but it literally matches the warning's
suggested rewrite from the pre-PR #51 code, so it's the lowest-risk
change.

### Fix 2 — go back to the blocking API and accept the warnings

Revert PR #51's changes to use the sync dict-style shorthand. Loses
the async-correctness wins, reintroduces the event-loop stalls, but
proves persistence works and isolates the problem.

Not a real fix — useful only as a bisection step if Fix 1 doesn't
work.

### Fix 3 — switch to `modal.Function`-based persistence

Define a dedicated Modal function with `max_containers=1` keyed on
guild_id, have it own the dict, and dispatch mutations through
`.remote()`. This is what `provision_workspace` does for filesystem
locks. Much heavier; only worth it if the dict-direct paths are
fundamentally broken for async callers.

Unlikely to be necessary. Listed for completeness.

## Validation plan

After applying Fix 1:

1. Restart the bot container (forces a fresh `RepoConfig` instance)
2. Run `/setrepo repo:leehanchung/SMILE-factory`
3. `modal dict items discord-orchestrator-repo-config` → confirm
   the entry is present
4. Restart the bot container again
5. `modal dict items discord-orchestrator-repo-config` → entry
   should STILL be there (this is the bug's core symptom — writes
   from a prior process should be visible to a new process)
6. `@delulu whats in this repo` → expect the `📁` subtitle and a
   real provisioned workspace, not the empty fast path
7. Repeat the test for `/admin_addrepo` on `RepoAllowlist` — same
   restart-across-a-bot-bounce test

## Why this is parked, not fixed now

The feature ships a degraded UX (users have to re-run `/setrepo`
after every bot restart) but does NOT ship broken behavior — the
refuse path is a clean re-bind, no data loss, no confusing state.
The async warning fix in PR #51 was the right architectural move
and shouldn't be reverted just to unblock the persistence path; the
right move is a surgical fix to the put-path semantics.

More importantly: the blocker was the module-path fix (PR #50),
which IS shipped. Users can drive the repo-provisioning end-to-end
in a single session. The persistence bug only bites on the second
session after a restart, which is annoying but recoverable. Ship
the feature, park the bug, fix in a follow-up when bandwidth
allows.

## Out of scope

- Migrating off `modal.Dict` to another storage backend
- Changing the schema of the stored binding
- Changing the key type (we use `channel_id: int`, which modal.Dict
  accepts; type mismatches are not the suspected issue)
- A write-through cache on `RepoConfig` to paper over the bug
  without fixing the underlying persistence — would hide real
  storage failures in the future
