# CLAUDE.md

LoRA-Instruct: LoRA fine-tuning for open-source causal LLMs using HuggingFace PEFT.

## Quick Reference

- **Setup**: `poetry install`
- **Lint**: `poetry run ruff check .` (fix with `--fix`)
- **Test**: `poetry run pytest`
- **Train**: `python finetune.py --base_model '<model>' --output_dir '<dir>'`

## Docs

- [Architecture](docs/architecture.md) — project structure, key components, and dependencies
- [Development](docs/development.md) — setup, linting, testing, training, and code conventions

## Git workflow

- Never commit to `main` — always branch, push, open a PR.
- Branch: `<type>/<slug>` (e.g. `feat/streaming-renderer`, `fix/dispatcher-timeout`).
- Commits + PR titles: [Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `build`, `perf`, `style`). Squash-merge uses the PR title, so it must be conventional.
- Only commit, push, or open PRs when explicitly asked. Never merge PRs — that's a human call.
