# Commit-back marker leaks + smoke-test eats error events

**Status:** bug-open

Two unrelated bugs in `delulu_sandbox_modal` that surfaced in a
Codex review of an in-flight PR. Captured here so they don't get
lost on the way to a real fix; neither is shipping behavior today
but both are real correctness issues.

This is a *bug report*, not a fix. Nothing in this document is in
the code yet.

## Source

Findings come from a Codex CLI review (run with `--sandbox
workspace-write`, which fell through to its MCP+web tools as
discussed in `prd/codex-ci-auth.md` §"bubblewrap on hosted
runners"). The review was posted on a PR whose number we didn't
record — but the file paths and line numbers are precise, so the
findings are reproducible by reading the code at those locations.

## Bug 1 — `.commit-branch` marker can leak into committed history

**Location**: `apps/delulu_sandbox_modal/src/delulu_sandbox_modal/repo_provisioner.py`,
function near the bare-cache + worktree provisioning logic.
Codex flagged line ~160; the surrounding helper that excludes
provisioning markers from the worktree tree only excludes the
`.provision.json` file written by the provisioning path.

### Symptom

`commit_workspace_changes` (the `/commit` Modal function in the
same module) writes a `.commit-branch` marker file *inside the
worktree* to remember which branch a thread is committing to.
Subsequent `/commit` invocations call `git add -A && git commit
-m ...` to stage and commit the user's changes — and `git add
-A` happily picks up `.commit-branch` along with everything
else, because nothing tells git to ignore it.

End result: a freshly-cleaned thread's `/commit` push includes
the bot's internal scratch file in the commit, then in the
pushed branch on GitHub. Visible to humans reviewing the PR;
visible in `git blame`. Not catastrophic but ugly and confusing
("what's `.commit-branch`?").

### Why it shipped

When the marker-exclusion code was written, `.provision.json`
was the only marker the sandbox wrote into a worktree. The
`/commit` flow added `.commit-branch` later as a separate
mechanism and didn't update the exclusion list (or, more
likely, didn't realize there was an exclusion list to update).

### Candidate fixes

Two reasonable shapes; pick one:

1. **Add `.commit-branch` to the marker-exclusion list.** Simplest
   fix; one line. Discoverable by future readers via the
   exclusion list itself. Downside: the marker still lives
   inside the worktree, so any third commit-back path that runs
   without the exclusion (e.g. a Claude `Bash: git add -A`
   tool call) would still re-introduce the leak.

2. **Move commit metadata outside the worktree.** Store
   `.commit-branch` in `$WORKSPACES_ROOT/.markers/<thread_id>/`
   or similar, alongside the workspace dir but not inside it.
   Bigger change but eliminates the leak path entirely. Also
   parallels how `.provision.json` *should* probably be stored
   (the provisioning step writes that one inside the worktree
   too — same risk if the user runs `git add` themselves).

Recommendation: start with (1) for the immediate fix, file (2)
as a parking-lot followup if more markers accumulate.

### Verification

After the fix, reproduce the original scenario:

1. `/setrepo` a test repo, `@delulu` a small change in the
   thread, `/commit "test"` — confirm `.commit-branch` is **not**
   in the pushed commit's tree (`git show <sha> --stat | grep -v
   commit-branch` should return everything).
2. `/commit "another"` — confirm the same.
3. Locally clone the test repo and `cat .commit-branch` —
   should fail with "no such file" (not in the working tree at
   all on a fresh clone).

## Bug 2 — Sandbox smoke test accepts error events as success

**Location**: `apps/delulu_sandbox_modal/tests/integration/test_sandbox_smoke.py`,
around line 40 — the part that consumes the event stream from
`run_claude_code.remote_gen()` and decides pass/fail.

### Symptom

The smoke test (intended as a daily E2E health check on the
deployed sandbox) treats terminal `error` events the same as
clean `done` events — it accepts the run as "completed" and
green-lights the test. So:

- Modal credentials expired → run yields an `error` event →
  smoke test passes.
- ChatGPT plan rate-limit → `error` event → smoke test passes.
- Sandbox image fails to pull → `error` event → smoke test
  passes.

The test workflow goes green while the deployed sandbox is
actually broken. The first signal of breakage is a real user
hitting it in Discord, not the daily check.

### Why it shipped

The smoke test was written defensively — error events were
treated as "the run finished, even if unhappily" so transient
failures wouldn't page anyone. That's the right instinct for an
*integration* test against external systems, but the threshold
got drawn in the wrong place: any `error` event is treated as
the same as a clean `done`, with no inspection of the error
payload.

### Candidate fixes

1. **Fail on any `error` event by default**, with an explicit
   opt-in for tests that exercise error handling. Simplest
   policy; matches "smoke test = is the happy path alive."
2. **Classify errors**: infra/credential/quota errors fail the
   test; explicit user-error or expected-cancel events pass.
   More accurate but bigger lift in the test framework.

Recommendation: (1) for the daily smoke; if there are tests
that genuinely need to assert specific error shapes, gate them
on a `pytest.mark.expects_error` (or similar) and only those
opt out.

### Verification

After the fix:

1. Force a known-bad credential into the smoke test's invocation
   (e.g. point `CLAUDE_CREDENTIALS_JSON` at an expired token in a
   one-off test job). Confirm the smoke test now **fails**, not
   passes.
2. Fix the credential. Confirm it goes back to passing.
3. Examine the daily Modal cron run history for a week to see
   how often a real `error` slips through; the answer should be
   zero. Anything else means the classifier (if we picked
   option 2) is letting things through.

## Out of scope (here, but possibly worth their own writeups)

- Auditing every `git add -A` call site in the bot for similar
  scratch-file leaks (Claude's bash tool, the eventual `/refresh`
  path, etc.). The marker-exclusion list is one of several layers
  of defense; this PRD only fixes the layer that bug 1 hit.
- Adding alerting on smoke-test green→red transitions. Even if
  the test is fixed to report errors honestly, no one notices a
  failed cron unless we hook it to a Discord/email channel.
- Splitting the smoke test into "is the sandbox alive at all"
  vs "can a real claude run complete." First is cheap and
  catches infra rot; second is expensive but catches model-side
  regressions.

## Effort estimate

- Bug 1 (option 1, exclusion list update): ~15 min including
  the verification steps. Bug 1 (option 2, move out of worktree):
  ~1 evening, since it touches every place that reads/writes the
  marker.
- Bug 2 (option 1, fail-on-error default): ~30 min if there are
  no existing tests that lean on error-event acceptance; longer
  if some have to be marked.
