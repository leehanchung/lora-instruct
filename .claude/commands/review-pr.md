---
description: Review a pull request
allowed-tools: Bash(gh api:*),Bash(gh pr diff:*),Bash(gh pr view:*),Write
---

# PR review — SMILE-factory

You are reviewing a pull request in the SMILE-factory monorepo.
Coordinate three specialist subagents, filter their output for
genuinely noteworthy findings, and submit one atomic review with
inline comments via the GitHub Reviews API.

## Repository context

The active project is the **delulu Discord orchestrator**:

- `apps/delulu_discord` — Discord bot, runs in Docker on a VPS.
  Changes here require a bot container rebuild.
- `apps/delulu_sandbox_modal` — Modal sandbox function that runs
  Claude Code. Changes here require a Modal redeploy.

The repo root also contains archived LoRA-Instruct fine-tuning code.
Lower review priority unless the PR explicitly touches it.

**Boundary invariants** (treat violations as correctness bugs):

- `delulu_discord` must never import from `delulu_sandbox_modal`.
- `delulu_sandbox_modal` must never import bot-side Discord types.

## Step 1 — Pre-flight

**Skip review** if any of these are true:

- The PR is a draft or closed
- The only changes are under `prd/` (planning docs)
- The only changes are documentation wording fixes

If skipping, there is no production code to object to, so the PR is
fine to merge as far as this reviewer is concerned. **Submit an
approving review** (not a bare comment) so it counts toward the
required approvals instead of leaving the PR stuck on "Review
required":
`gh api repos/{owner}/{repo}/pulls/<N>/reviews --method POST -f event=APPROVE -f body="Approving — skipped detailed review (<reason>; prd/docs only)."`
If that fails with "cannot approve your own pull request", retry once
with `event=COMMENT`. Then stop.

## Step 2 — Gather context

Run `gh pr view <N>` and `gh pr diff <N>` to collect metadata,
title, body, and the full diff. Note any `CLAUDE.md` files at the
repo root and in directories touched by the PR — they define
conventions to check against.

## Step 3 — Delegate to subagents

Launch all three subagents **in parallel**, passing each the PR
diff and relevant context:

1. **code-quality-reviewer** — correctness, boundary invariants,
   clean code
2. **security-reviewer** — credential handling, injection,
   sandbox blast radius
3. **test-coverage-reviewer** — missing tests, untested paths

Each subagent returns a list of findings. Only provide noteworthy
feedback — do not speculate or pad.

## Step 4 — High-signal filter

From the combined subagent output, **only keep findings that clear
all of these bars:**

- The code will fail to compile, parse, or run correctly
  **regardless of inputs**, OR
- There is a **clear, quotable violation** of a rule in a scoped
  `CLAUDE.md` file, OR
- The change introduces a **specific, demonstrable security
  regression**

**Filter OUT:**

- Style nits that ruff would catch (ruff runs in pre-commit)
- Documentation wording unless factually wrong
- General "code quality" concerns
- Potential issues that depend on unknown inputs or external state
- Subjective suggestions
- Pre-existing issues on the base branch

If you are not certain an issue is real, **do not include it.**
False positives erode trust.

## Step 5 — Write the review payload

Use the `Write` tool for both files. **Do not use shell heredoc.**

### 5a — Summary body (`/tmp/review.md`)

```markdown
## Review summary

<one-sentence verdict: "LGTM" or "minor non-blocking notes" (both
approve) — or "flagging N issue(s) for human review" (comment)>

## Deployment

<which side(s) need a redeploy/rebuild, if any. Skip if the PR
doesn't touch either app.>
```

### 5b — Inline comments (`/tmp/review_comments.json`)

```json
{
  "comments": [
    {
      "path": "relative/file/path.py",
      "line": 42,
      "body": "**Correctness:** Brief description of the issue."
    }
  ]
}
```

- `path` — relative to repo root, must match the diff path exactly.
- `line` — line number in the **new version** of the file. Must be
  a line visible in the diff (added or context line within a hunk).
  For ranges, use the last line.
- `body` — prefix with the category. Keep it specific and actionable.

If there are no findings, write `{"comments": []}`.

## Step 6 — Submit exactly one review

**Advisory only — never `REQUEST_CHANGES`.**

**Default to APPROVE.** `main` requires approving reviews, so a clean
review that only leaves a COMMENT forces a needless human
self-approval. Decide the event from your *findings*, not your tone:

- **APPROVE** — `/tmp/review_comments.json` has zero inline findings
  **and** the body raises no blocking concern. "LGTM" and "minor
  non-blocking notes" both approve.
- **COMMENT** — there is at least one inline finding, **or** you are
  putting a substantive concern a human must weigh in the body.

Submit with a **single `gh api` command** that starts with `gh api`
and passes the summary by file reference. This is load-bearing: the
sandbox only permits `Bash(gh api:*)` and rejects any command it can't
statically analyze — so do **NOT** prefix the command with a variable
assignment (`REVIEW_BODY=...`), and do **NOT** use command
substitution (`$(cat ...)`) or a `jq`/`cat` pipeline. Those forms are
silently blocked and the review never posts.

Decide the event yourself per the rule above and write it as a literal
(`APPROVE` when `/tmp/review_comments.json` is `{"comments": []}` and
nothing blocking is in the body; otherwise `COMMENT`):

```bash
gh api "repos/{owner}/{repo}/pulls/<N>/reviews" --method POST -f event=APPROVE -F body=@/tmp/review.md --input /tmp/review_comments.json --jq '.html_url'
```

- `-F body=@/tmp/review.md` reads the summary from the file (no shell
  quoting, nothing to statically analyze).
- `--input /tmp/review_comments.json` attaches the inline comments.
- Swap `event=APPROVE` → `event=COMMENT` when there are findings.

**One attempt, one outcome — never retry**, with one exception: if
`APPROVE` fails with "cannot approve your own pull request", retry once
with `event=COMMENT`.
