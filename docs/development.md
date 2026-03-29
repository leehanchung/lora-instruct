# Development Guide

## Setup

```bash
poetry install
```

## Linting

Ruff is configured in `pyproject.toml` with rules: E, W, F, I, C, B (ignoring E501, B008, C901).

```bash
poetry run ruff check .
poetry run ruff check --fix .
```

## Testing

```bash
poetry run pytest
```

Note: test coverage is configured in `tox.ini` but currently commented out.

## Training

```bash
python finetune.py --base_model 'togethercomputer/RedPajama-INCITE-Base-7B-v0.1' --output_dir './lora-redpajama'
```

All training hyperparameters are CLI flags — see `train()` function signature in `finetune.py` for the full list.

### Distributed Training

```bash
export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=1234 finetune.py \
    --base_model 'togethercomputer/RedPajama-INCITE-Base-7B-v0.1' \
    --output_dir './lora-redpajama'
```

## Inference Benchmarking

```bash
cd inference
python bench.py --api-url https://api.openai.com/v1/chat/completions --model gpt-3.5-turbo
```

Requires `OPENAI_API_KEY` in environment or `.env` file.

## Code Conventions

- Python 3.10+
- Type hints on function signatures
- Dataclasses for structured configs (see `TrainConfig`)
- `python-fire` for CLI interfaces
- `python-dotenv` for environment variable loading
