# 🌲🤏 LoRA-Instruct

This repository contains code for instruction fine-tuning permissive open source LLMs using [low-rank adaptation (LoRA)](https://arxiv.org/abs/2106.09685). Code is tested using [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) dataset.

- Estimated training time for fine-tuning RedPajama-INCITE-Base-7B-v0.1 with a single RTX 3090 and Stanford Alpaca is ~12 hours.
- Estimated training time for fine-tuning RedPajama-INCITE-Base-7B-v0.1 with RTX 3090 and RTX Titan and Stanford Alpaca is ~6.5 hours.
- Currently only supports LoRA Instruct fine-tuning [RedPajama-INCITE-Base-7B-v0.1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1).


Inspired by [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)

# Known Models
| Model | 3B | 7B | 13B |
|:-------|:----:|:----:|:-----:|
| LLaMA | :white_large_square: | :white_large_square: | :white_large_square: |
| INCITE-RedPajama | :white_large_square: | :white_check_mark: | :white_large_square: |
| MPT | :white_large_square: | :white_large_square: | :white_large_square: |

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

#### Hardware Spec
```
Ubuntu 20.04.1 LTS on Windows 11 WSL

Driver Version: 531.41
CUDA Version: 12.1
cuDNN version: 8.5.0
```

## References
- [LoRA: Low-Rank Adapation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [State-of-the-art Parameter-Efficient Fine-Tuning (PEFT) methods](https://github.com/huggingface/peft)
- [Low-rank Adaption of Large Language Models: Explaining the Key Concepts Behind LoRA](https://www.youtube.com/watch?v=dA-NhCtrrVE) by [Chris Alexiuk](https://www.linkedin.com/in/csalexiuk/)
- [Low-rank Adaption of Large Language Models Part 2: Simple Fine-tuning with LoR ](https://www.youtube.com/watch?v=iYr1xZn26R8) by [Chris Alexiuk](https://www.linkedin.com/in/csalexiuk/)