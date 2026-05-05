---
description: Review a pull request for the SMILE-factory monorepo. Produces exactly one advisory PR review (approve or comment, never request-changes) with inline comments on specific lines via the GitHub Reviews API.
allowed-tools: Write, WebFetch, Bash(gh pr review:*), Bash(gh api:*)
---

# PR review — SMILE-factory

You are reviewing a pull request in the SMILE-factory monorepo. This
command is adapted from the `code-review` plugin in
`anthropics/claude-code` (the canonical high-signal PR reviewer),
with three project-specific adaptations:

1. Scoped to the delulu Discord orchestrator architecture and its
   boundary invariants (see below).
2. **Approve-or-comment only.** Never `--request-changes`.
3. Submits exactly one formal review via the GitHub Reviews API with
   **inline comments on specific diff lines** plus a top-level
   summary body. Our workflow wires the review into GitHub's
   Reviewers box so branch protection can gate on it.

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
tree — don't enforce `apps/delulu_discord/CLAUDE.md` rules against
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

### Step 5 — Write the review payload

Build two artifacts using the `Write` tool. **Do not use shell
heredoc (`cat > file <<EOF`)** — markdown with code fences and
backticks collides with bash quoting, which has caused
duplicate-review pollution in the past.

#### 5a — Summary body (`/tmp/review.md`)

```markdown
## Review summary

<one-sentence verdict: "LGTM", "minor observations", or
"flagging N issues for human review">

## Deployment

<which side(s) need a redeploy/rebuild, if any. Skip this section
if the PR doesn't touch either app.>
```

Keep the body concise. Do not manufacture content to fill space.

#### 5b — Inline comments (`/tmp/review_comments.json`)

A JSON object with a `comments` array. Each comment has:

```json
{
  "comments": [
    {
      "path": "relative/file/path.py",
      "line": 42,
      "body": "**Priority 1 — Correctness:** Brief description of the issue."
    }
  ]
}
```

- `path` — file path relative to repo root (must match the path in
  the diff exactly).
- `line` — the line number in the **new version** of the file where
  the comment should appear. Must be a line that is part of the
  diff hunk (added or context line visible in the diff). If the
  issue spans a range, use the last line of the range.
- `body` — the comment text. Prefix with the priority category
  (e.g. "**Priority 1 — Correctness:**"). Keep it specific and
  actionable.

If there are no findings, write `{"comments": []}`.

**Line number rules:**
- Only reference lines that appear in the diff (added lines or
  unchanged context lines within a hunk). GitHub will reject
  comments on lines outside the diff.
- Use the line number from the new file (right side of the diff),
  not the old file.

### Step 6 — Submit exactly one review

**Your review is advisory, not blocking.**

Read `/tmp/review.md` into a shell variable for the body, then
submit via the GitHub Reviews API using `gh api`. This single call
posts both the top-level summary and all inline comments atomically.

Pick the event based on findings:

- No substantive issues → `event=APPROVE`
- Issues / observations / questions → `event=COMMENT`

**Do not use `REQUEST_CHANGES`.** A request-changes review gives
your individual judgment a hard veto over merges, and you can be
wrong. High-confidence findings still go in the comment body — the
human reviewer decides whether they block merge.

Submit using `gh api`:

```bash
REVIEW_BODY=$(cat /tmp/review.md)
gh api \
  "repos/{owner}/{repo}/pulls/<N>/reviews" \
  --method POST \
  -f event="APPROVE" \
  -f body="$REVIEW_BODY" \
  --input /tmp/review_comments.json \
  --jq '.html_url'
```

`gh api --input` sends the JSON file as the request body, and the
`-f` flags are merged on top. The final request contains `event`,
`body`, and `comments`.

**Never retry on apparent failure.** One attempt, one outcome.
Duplicate reviews pollute the PR history.

**The only permitted fallback:** if submitting with `APPROVE` fails
with a "cannot approve your own pull request" error, fall back to
the same call with `event=COMMENT`. One retry maximum, only for
this specific error.
