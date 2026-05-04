# LoRA-Instruct

LoRA fine-tuning recipe for permissive open-source causal LLMs, built on
HuggingFace `transformers` + `peft`. This project applies
[low-rank adaptation (LoRA)](https://arxiv.org/abs/2106.09685) to decoder-only
models for instruction tuning, and is tested against the
[Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) dataset.

Inspired by [Alpaca-LoRA](https://github.com/tloen/alpaca-lora).

## Quick Start

Run everything from this directory:

```bash
cd training/lora_instruct/
poetry install
```

Train a LoRA adapter on top of a base model:

```bash
poetry run python finetune.py \
    --base_model 'togethercomputer/RedPajama-INCITE-Base-7B-v0.1' \
    --output_dir './lora-redpajama'
```

Default dataset is `yahma/alpaca-cleaned`; override with `--data_path`. See
`TrainConfig` in `finetune.py` for the full set of CLI flags (batch size,
LoRA rank, dropout, target modules, prompt template, etc.).

To fine-tune on an NVIDIA 2000-series GPU or earlier, comment out the
following line in `finetune.py`:

```python
model = prepare_model_for_int8_training(model)
```

## Distributed Training (HuggingFace Accelerate)

For multi-GPU training, set the world size and visible devices, then launch
with `torchrun`:

```bash
export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1

torchrun \
    --nproc_per_node=2 \
    --master_port=1234 \
    finetune.py \
    --base_model 'togethercomputer/RedPajama-INCITE-Base-7B-v0.1' \
    --output_dir './lora-redpajama'
```

## Trained Models

| Model        | Runs | Training Time | Link |
|:-------------|:----:|:-------------:|:----:|
| LLaMA 3B     |      |               |      |
| LLaMA 7B     |      |               |      |
| RedPajama 3B | yes  | 1:44:14       |      |
| RedPajama 7B | yes  | 3:09:58       |      |
| MPT 3B       |      |               |      |
| MPT 7B       |      |               |      |
| Falcon 7B    | yes  |               |      |

Estimated wall-clock for fine-tuning RedPajama-INCITE-Base-7B-v0.1 with
Stanford Alpaca:

- ~12 hours on a single RTX 3090.
- ~6.5 hours on RTX 3090 + RTX Titan.

### Training Hardware Spec

```
Ubuntu 20.04.1 LTS (WSL2)

Driver Version: 531.41
CUDA Version: 12.1
cuDNN version: 8.5.0
```

## Where things live

- `finetune.py` — entry point. Straightforward application of PEFT / LoRA to
  a decoder-only model, plus prompt construction and tokenization glue.
- `utils/` — `prompter.py` (template-driven prompt builder) and training
  callbacks.
- `templates/` — prompt templates loaded by `Prompter`. Default is
  `alpaca.json`.
- `dataset/` — bundled instruction datasets (`alpaca_data.json`,
  `alpaca_data_cleaned_archive.json`, `alpaca_data_gpt4.json`,
  `prompts.jsonl`).
- `inference/` — local inference + benchmarking (`bench.py`,
  `deploy_local.sh`).
- `notebook/` — per-base-model exploratory notebooks (LLaMA-7B, MPT-7B,
  RedPajama-INCITE-7B).

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [State-of-the-art Parameter-Efficient Fine-Tuning (PEFT) methods](https://github.com/huggingface/peft)
- [Stanford Alpaca: An Instruction-following LLaMA Model](https://github.com/tatsu-lab/stanford_alpaca)
- [EMNLP 2022 Tutorial: Modular and Parameter-Efficient Fine-Tuning for NLP Models](https://www.youtube.com/watch?v=KoOlcX3XLd4)
