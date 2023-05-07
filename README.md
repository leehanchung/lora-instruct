# 🌲🤏 LoRA-Instruct

This repository contains code for instruction fine-tuning permissive open source LLMs using [low-rank adaptation (LoRA)](https://arxiv.org/abs/2106.09685).

- Currently only supports LoRA Instruct fine-tuning [RedPajama-INCITE-Base-7B-v0.1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1).



Inspired by [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)

### Local Setup

Install dependencies

```bash
poetry install
```

### Training (`finetune.py`)

This file contains a straightforward application of PEFT / LoRA to decoder only model,
as well as some code related to prompt construction and tokenization.

Example usage:
```bash
python finetune.py \
    --base_model 'togethercomputer/RedPajama-INCITE-Base-7B-v0.1' \
    --output_dir './lora-redpajama'
```

#### Training on 2 GPUs.

* NOTE: please set the following environment variables
```bash
export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1
```

```bash
torchrun \
    --nproc_per_node=2 \
    --master_port=1234 \
    finetune.py \
    --base_model 'togethercomputer/RedPajama-INCITE-Base-7B-v0.1' \
    --output_dir './lora-redpajama'
```
