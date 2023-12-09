# :teacher:🤏 LoRA-Instruct

This repository contains code for fine-tuning permissive open source LLMs using [low-rank adaptation (LoRA)](https://arxiv.org/abs/2106.09685).

Code is tested using [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) dataset.

- Estimated training time for fine-tuning RedPajama-INCITE-Base-7B-v0.1 with a single RTX 3090 and Stanford Alpaca is ~12 hours.
- Estimated training time for fine-tuning RedPajama-INCITE-Base-7B-v0.1 with RTX 3090 and RTX Titan and Stanford Alpaca is ~6.5 hours.
- Currently only supports LoRA Instruct fine-tuning [RedPajama-INCITE-Base-7B-v0.1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1).


Inspired by [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)

## Trained Models
| Model | Runs | Training Time  | Link |
|:-------|:----:|:----:|:-----:|
| LLaMA 3B | :white_large_square: |  |  |
| LLaMA 7B | :white_large_square: |  |  |
| RedPajama 3B | :white_check_mark: | 1:44:14 | |
| RedPajama 7B | :white_check_mark: | 3:09:58 | |
| MPT 3B | :white_large_square: |  |  |
| MPT 7B | :white_large_square: |  |  |

#### Training Hardware Spec
```
Ubuntu 20.04.1 LTS (WSL2)

Driver Version: 531.41
CUDA Version: 12.1
cuDNN version: 8.5.0
```

### Local Setup

Install dependencies
```bash
poetry install
```

To fine-tune using NVidia 2000 series GPU or earlier, please comment out this line in `finetune.py`
```python
model = prepare_model_for_int8_training(model)
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

#### Distributed Training with 🤗 Accelerate

We uses HuggingFace's `accelerate` library for distributed training. The following is an example for distributed training with two GPUs.

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

## References
- [LoRA: Low-Rank Adapation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [State-of-the-art Parameter-Efficient Fine-Tuning (PEFT) methods](https://github.com/huggingface/peft)
- [Stanford Alpaca: An Instruction-following LLaMA Model](https://github.com/tatsu-lab/stanford_alpaca)
- [EMNLP 2022 Tutorial: Modular and Parameter-Efficient Fine-Tuning for NLP Models](https://www.youtube.com/watch?v=KoOlcX3XLd4)