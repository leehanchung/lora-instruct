# CLAUDE.md

SMILE-factory is a multi-project monorepo. The active project is **delulu**, a Discord → Claude Code orchestrator that dispatches tasks to ephemeral Modal sandboxes (under `apps/delulu_discord` and `apps/delulu_sandbox_modal`). It also hosts a **LoRA-Instruct** SFT training recipe (HuggingFace PEFT + LoRA) under `training/lora_instruct/`.

## Quick Reference

Each project owns its own Makefile and tooling — there is no repo-wide `poetry`/`pytest`. Use the per-project entrypoints:

- **Delulu (active)**: `make -C apps/delulu_discord check` (lint + test), `make -C apps/delulu_discord deploy`, `make -C apps/delulu_sandbox_modal modal-deploy`. Top-level shortcuts: `make check`, `make deploy-bot`, `make deploy-modal`, `make deploy-all`.
- **LoRA-Instruct training**: see [`training/lora_instruct/CLAUDE.md`](training/lora_instruct/CLAUDE.md) for setup, lint, test, and training commands — they no longer live at the repo root.

## Working in the monorepo

The repo is category-first: `apps/` (production services), `training/`
(reusable SFT/RLHF recipes), `data/` (cross-project datasets + scrapers),
`infra/` (shared infra, e.g. the managed-agents platform), plus `docs/`,
`prd/`, and an empty `libs/` slot for future cross-project packages. Most
tasks are scoped to a single project — work from that project's directory,
not the repo root:

- **Identify the project first.** Almost every task lives inside one
  project (`apps/delulu_discord`, `training/lora_instruct`, …).
- **Work from the project root**, not the repo root — `cd` in or use
  `make -C <project>`. Each project owns its deps, lockfile, and Makefile.
- **Scope `Grep`/`Glob` to the project dir**; broaden only for genuinely
  cross-cutting work.
- **No imports between projects.** Shared code goes in `libs/` via its own PR.
- **One PR per project.** Cross-cutting files (`.github/`, root docs, `prd/`,
  `.pre-commit-config.yaml`) ship as their own PR.
- **A project's own `CLAUDE.md` wins** over this file for its commands and
  conventions.

## Docs

- [Monorepo conventions](docs/monorepo-conventions.md) — how pre-commit, CI, branch protection, and per-app Makefiles interact in this multi-pyproject layout. Read this before touching `.pre-commit-config.yaml`, `.github/workflows/delulu-deploy.yaml`, or any root-level tooling
- [Architecture](docs/architecture.md) — project structure, key components, and dependencies
- [Development](docs/development.md) — setup, linting, testing, training, and code conventions

## Git workflow

- Never commit to `main` — always branch, push, open a PR.
- Branch: `<type>/<slug>` (e.g. `feat/streaming-renderer`, `fix/dispatcher-timeout`).
- Commits + PR titles: [Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `build`, `perf`, `style`, plus a local `prd` type for changes under any `prd/` directory — root or per-app). Squash-merge uses the PR title, so it must be conventional.
- Only commit, push, or open PRs when explicitly asked. Never merge PRs — that's a human call.
