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

If skipping, post a one-line "skipped — reason" comment via
`gh api repos/{owner}/{repo}/pulls/<N>/reviews --method POST -f event=COMMENT -f body="Skipped — <reason>"`.
Then stop.

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

<one-sentence verdict: "LGTM", "minor observations", or
"flagging N issues for human review">

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

- No substantive issues → `event=APPROVE`
- Issues or observations → `event=COMMENT`

**Never retry on failure.** One attempt, one outcome.

**Only fallback:** if `APPROVE` fails with "cannot approve your own
pull request", retry once with `event=COMMENT` using the same body.
