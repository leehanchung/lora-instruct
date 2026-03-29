# Architecture

## Project Overview

LoRA-Instruct fine-tunes open-source causal LLMs using Low-Rank Adaptation (LoRA) via HuggingFace PEFT. It also includes an inference benchmarking tool for OpenAI-compatible endpoints.

## Directory Structure

```
├── finetune.py              # Main training entrypoint (uses python-fire CLI)
├── utils/
│   ├── prompter.py          # Prompt template loading and formatting
│   └── callbacks.py         # Streaming generation helpers (Stream, Iteratorize)
├── inference/
│   ├── bench.py             # Async LLM endpoint benchmarking tool
│   ├── deploy_local.sh      # Local deployment script
│   └── sonnet.txt           # Sentence bank for benchmark prompts
├── templates/
│   └── alpaca.json          # Alpaca prompt template (prompt_input / prompt_no_input)
├── dataset/                 # Training datasets (Alpaca variants, prompts.jsonl)
├── notebook/                # Jupyter notebooks for model experimentation
├── pyproject.toml           # Poetry project config and dependencies
└── tox.ini                  # Tox test environments (linting, formatting)
```

## Key Components

### Fine-tuning (`finetune.py`)

- **Entry point**: `fire.Fire(train)` — all training params are CLI flags.
- **Model loading**: `AutoModelForCausalLM` with bfloat16, optional 8-bit quantization via BitsAndBytes.
- **LoRA**: Applied via `peft.LoraConfig` with configurable rank, alpha, dropout, and target modules (default: `["up_proj"]`).
- **Data**: Loads from HuggingFace Hub or local JSON/JSONL files.
- **Distributed training**: Supports DDP via `torchrun` with `WORLD_SIZE` and `LOCAL_RANK` env vars.
- **`TrainConfig` dataclass**: Structured config (currently used by `setup_model()`, but `train()` function uses flat kwargs).
- **`TokenizerHelper`**: Handles prompt tokenization and optional input masking (`train_on_inputs=False`).

### Prompt System (`utils/prompter.py`)

- `Prompter` loads JSON templates from `templates/` directory.
- Templates define `prompt_input`, `prompt_no_input`, and `response_split` fields.
- Currently only `alpaca.json` template exists.

### Inference Benchmarking (`inference/bench.py`)

- Async HTTP benchmarking for OpenAI-compatible chat completion endpoints.
- Uses Poisson process for request arrival simulation.
- Measures: TTFT, end-to-end latency, inter-token delay, tokens/s.
- Validates responses using embedded random numeric keys.

## Dependencies

Core: `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `torch`
Inference: `httpx`, `tiktoken`, `numpy`, `pandas`, `num2words`
Dev: `ruff`, `pytest`, `ipykernel`
