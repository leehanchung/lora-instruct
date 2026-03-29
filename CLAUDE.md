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
