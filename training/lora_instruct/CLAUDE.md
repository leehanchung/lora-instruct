# CLAUDE.md — training/lora_instruct

LoRA SFT training recipe for fine-tuning open-source causal LLMs with HuggingFace PEFT.

## Quick Reference

This project is run from its own directory, not the repo root:

```bash
cd training/lora_instruct/
```

- **Setup**: `poetry install`
- **Lint**: `poetry run ruff check .` (fix with `--fix`)
- **Test**: `poetry run pytest`
- **CLI help**: `poetry run python finetune.py --help`
- **Train**: `poetry run python finetune.py --base_model '<model>' --output_dir '<dir>'`

## Notes

- This project still uses **poetry** (`pyproject.toml` + `poetry.lock`). The
  apps under `apps/` use `uv` — do not confuse the two.
- `finetune.py` reads its training config from CLI flags via `fire`. See
  `TrainConfig` at the top of `finetune.py` for the full list of knobs
  (batch size, LoRA rank, target modules, prompt template, etc.).
- Default dataset is `yahma/alpaca-cleaned`; default prompt template is
  `alpaca` (see `templates/alpaca.json`).
- Distributed training uses HuggingFace `accelerate` / `torchrun` — see the
  README for the multi-GPU invocation.

## Layout

- `finetune.py` — entry point, PEFT/LoRA training loop.
- `utils/prompter.py` — prompt construction from templates.
- `templates/` — prompt templates (alpaca by default).
- `dataset/` — bundled instruction datasets (alpaca, gpt4 variants).
- `inference/` — local inference + benchmarking scripts.
- `notebook/` — exploratory notebooks per base model.
- `tests/` — lightweight pytest suite (prompter, template structure,
  dataset file integrity). Heavy ML imports are out of scope —
  these run without GPU or transformers/peft installed.
