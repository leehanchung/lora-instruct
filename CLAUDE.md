# CLAUDE.md

LoRA-Instruct: LoRA fine-tuning for open-source causal LLMs using HuggingFace PEFT.

## Quick Reference

- **Setup**: `poetry install`
- **Lint**: `poetry run ruff check .` (fix with `--fix`)
- **Test**: `poetry run pytest`
- **Train**: `python finetune.py --base_model '<model>' --output_dir '<dir>'`

## Docs

- [Monorepo conventions](docs/monorepo-conventions.md) — how pre-commit, CI, branch protection, and per-app Makefiles interact in this multi-pyproject layout. Read this before touching `.pre-commit-config.yaml`, `.github/workflows/delulu-deploy.yaml`, or any root-level tooling
- [Architecture](docs/architecture.md) — project structure, key components, and dependencies
- [Development](docs/development.md) — setup, linting, testing, training, and code conventions

## Git workflow

- Never commit to `main` — always branch, push, open a PR.
- Branch: `<type>/<slug>` (e.g. `feat/streaming-renderer`, `fix/dispatcher-timeout`).
- Commits + PR titles: [Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `build`, `perf`, `style`, plus a local `prd` type for changes under `prd/`). Squash-merge uses the PR title, so it must be conventional.
- Only commit, push, or open PRs when explicitly asked. Never merge PRs — that's a human call.
