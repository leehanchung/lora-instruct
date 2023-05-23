import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(base_model: str, lora_weights: str, load_8bit: bool = True,):
    # if device == "cuda:o":
    print(torch.cuda.is_available())
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )
    print(model.eval())

    input = "this is a good day to die"
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    input_ids = tokenizer(input, return_tensors="pt").input_ids.to(device)

    output = model.generate(
        input_ids=input_ids,
        # generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=500,
    )
    print(output)
    # else:
    #     print('fuckkkkk')


if __name__ == "__main__":
    fire.Fire(main)
