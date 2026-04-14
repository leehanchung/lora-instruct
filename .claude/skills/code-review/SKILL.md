---
name: code-review
description: Review a pull request for the SMILE-factory monorepo. Produces exactly one advisory PR review (approve or comment, never request-changes) via `gh pr review --body-file`. Mirrors the multi-step, high-signal approach from anthropics/claude-code's code-review plugin but scoped to the delulu Discord orchestrator architecture.
---

# PR review — SMILE-factory

You are reviewing a pull request in the SMILE-factory monorepo. The
process below is adapted from the `code-review` plugin in
`anthropics/claude-code` (the canonical high-signal PR reviewer),
with three project-specific adaptations:

1. Scoped to the delulu Discord orchestrator architecture and its
   boundary invariants (see below).
2. **Approve-or-comment only.** Never `--request-changes`.
3. Submits exactly one top-level formal review via `gh pr review
   --body-file`, rather than posting inline comments. Our workflow
   wires the review into GitHub's Reviewers box so branch protection
   can gate on it.

## Repository context

The active project is the **delulu Discord orchestrator**:

- `apps/delulu_discord` — Discord bot, runs in Docker on a VPS.
  Changes here require a bot container rebuild.
- `apps/delulu_sandbox_modal` — Modal sandbox function that runs
  Claude Code. Changes here require a Modal redeploy.

The repo root also contains archived LoRA-Instruct fine-tuning code.
Lower review priority unless the PR explicitly touches it.

**Boundary invariants that the code must respect** (treat violations
as correctness issues):

- `delulu_discord` must never import from `delulu_sandbox_modal`
  (deployed separately).
- `delulu_sandbox_modal` must never import bot-side Discord types.

## Review process

### Step 1 — Pre-flight

Decide whether this PR should be reviewed at all. **Skip review** if
any of these are true:

- The PR is a draft or closed
- The only changes are under `prd/` (planning docs, not production code)
- The only changes are documentation wording fixes

If skipping, write a one-line "skipped — reason" note to
`/tmp/review.md` and submit a single `gh pr review <N> --comment
--body-file /tmp/review.md`. Then stop.

### Step 2 — Load PR context

Fetch PR metadata and the diff via WebFetch against the GitHub REST
API (not via bash, not via local git):

- `https://api.github.com/repos/<owner>/<repo>/pulls/<N>` for
  metadata (title, body, author, base/head SHAs)
- `https://api.github.com/repos/<owner>/<repo>/pulls/<N>/files` for
  the file list and patches

Note any `CLAUDE.md` files at the repo root and in directories
touched by the PR. They define project conventions you should check
against. A `CLAUDE.md` file only applies to files in its directory
tree — don't enforce a `apps/delulu_discord/CLAUDE.md` rule against
a `apps/delulu_sandbox_modal/` file.

### Step 3 — Review in priority order

Evaluate the diff in this order. Only flag real, high-confidence
issues. Do not speculate.

**Priority 1 — Correctness.** Actual bugs that will break the code.
Focus on:

- Compile/parse errors: syntax errors, type errors, missing imports,
  unresolved references
- Clear logic errors that produce wrong results regardless of inputs
- Missing error handling at system boundaries: Discord gateway
  callbacks, Modal dispatch, subprocess exec, async generator
  cleanup
- Async/await misuse (e.g., `asyncio.get_event_loop()` inside
  `async def` when `get_running_loop()` is correct)

**Priority 2 — Security.** Credential handling, path traversal in
the attachment-write path, subprocess argument injection, anything
that widens the Modal sandbox's blast radius.

**Priority 3 — Boundary invariants.** Cross-boundary imports between
the bot and the sandbox. Violations are correctness issues.

**Priority 4 — Deployment impact.** If the PR touches
`apps/delulu_sandbox_modal/`, a Modal redeploy is needed. If it
touches `apps/delulu_discord/`, a bot container rebuild is needed.
Flag if the PR description or commit messages don't mention which
side is affected when the change spans both.

**Priority 5 — Test coverage.** Note missing tests but **do not
block on them** — the project has minimal test infrastructure today.

### Step 4 — High-signal filter

**Only keep issues that clear all of these bars:**

- The code will fail to compile, parse, or run correctly **regardless
  of inputs**, OR
- There is a **clear, quotable violation of a rule** in a scoped
  `CLAUDE.md` file, OR
- The change introduces a **specific, demonstrable security
  regression**

**Filter OUT:**

- Style nits that ruff would catch — ruff runs in pre-commit
- Documentation wording unless factually wrong
- General "code quality" concerns
- Potential issues that depend on unknown inputs or external state
- Subjective suggestions and improvements
- Pre-existing issues (things that already exist on the base branch)

If you are not certain an issue is real, **do not flag it.** False
positives erode trust and waste reviewer time.

### Step 5 — Write the review body

Use the `Write` tool to save the review body to `/tmp/review.md`.
**Do not use shell heredoc (`cat > file <<EOF`)** — markdown with
code fences and backticks collides with bash quoting, which has
caused duplicate-review pollution in the past.

Body format:

```markdown
## Review summary

<one-sentence verdict: "LGTM", "minor observations", or
"flagging N issues for human review">

## Findings

<list of high-signal issues grouped by priority. Only include this
section if there are findings. Each finding should include file +
line range, the issue, and why it's high-signal. Skip this section
entirely if no findings.>

## Deployment

<which side(s) need a redeploy/rebuild, if any. Skip this section
if the PR doesn't touch either app.>
```

Keep the body concise. Do not manufacture content to fill space.

### Step 6 — Submit exactly one review

**Your review is advisory, not blocking.** Pick exactly one of:

- No substantive issues →
  `gh pr review <N> --approve --body-file /tmp/review.md`
- Issues / observations / questions →
  `gh pr review <N> --comment --body-file /tmp/review.md`

**Do not use `--request-changes`.** A request-changes review gives
your individual judgment a hard veto over merges, and you can be
wrong. High-confidence findings still go in the comment body — the
human reviewer decides whether they block merge.

**Never retry on apparent failure.** One attempt, one outcome.
Duplicate reviews pollute the PR history.

**The only permitted fallback:** if `gh pr review <N> --approve`
fails with a "cannot approve your own pull request" error, fall
back to `gh pr review <N> --comment --body-file /tmp/review.md`
with the same body file. One retry maximum, only for this specific
error.
