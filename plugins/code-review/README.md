# code-review plugin

A Claude Code plugin that performs an advisory PR code review scoped
to the SMILE-factory monorepo.

## What it does

Invokes `/code-review:code-review <repo>/pull/<N>` and produces
**exactly one** formal GitHub PR review via `gh pr review
--body-file`. Outcomes are limited to:

- `--approve` — no substantive issues
- `--comment` — anything else (observations, concerns, questions,
  high-confidence bugs the human should confirm)

**Never `--request-changes`.** A model's judgment shouldn't hard-
veto merges; the human decides what blocks.

## Where it's used

Called from `.github/workflows/claude-code-review.yml` on every
`pull_request` event (excluding drafts and PRs authored by
`claude[bot]`). See that workflow file for the exact
`plugin_marketplaces` / `plugins` / `prompt` wiring.

You can also invoke it interactively if you run `claude` against
this repo:

```
/code-review:code-review leehanchung/SMILE-factory/pull/42
```

## Review scope

See `commands/code-review.md` for the full prompt. Highlights:

- **Priority 1** — Correctness (compile errors, logic errors, async
  misuse, boundary error handling)
- **Priority 2** — Security (credential handling, path traversal,
  subprocess argument injection, sandbox blast radius)
- **Priority 3** — Boundary invariants (no cross-imports between
  `delulu_discord` and `delulu_sandbox_modal`)
- **Priority 4** — Deployment impact (Modal redeploy / bot
  container rebuild when touching the apps)
- **Priority 5** — Test coverage (noted but non-blocking)

High-signal filter explicitly excludes style nits, pre-existing
issues, subjective suggestions, and anything that depends on
unknown inputs or external state.

## Editing this plugin

One gotcha: the GitHub Action fetches this plugin from the
**default branch** of the repo URL passed to `plugin_marketplaces`.
When `plugin_marketplaces` points at this repo, **edits to this
plugin take effect only after they land on `main`** — a PR that
edits the plugin won't exercise its own changes. Same
self-validation caveat as the workflow files themselves.

To test a plugin edit: merge it to `main` (bypass any blocked
checks if needed, the plugin is the thing being tested), then
exercise the new behavior on a follow-up PR.
