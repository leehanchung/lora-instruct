import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import fire
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import logger as tokenization_logger

from utils.prompter import Prompter

load_dotenv()
warnings.filterwarnings(
    "ignore",
    message=".*GPTNeoXTokenizerFast.*",
    category=UserWarning,
    module="transformers.tokenization_utils_base",
)
tokenization_logger.setLevel("ERROR")
torch.cuda.empty_cache()


@dataclass
class TrainConfig:
    base_model: str
    data_path: str = "yahma/alpaca-cleaned"
    output_dir: str = "./lora-alpaca"
    device_map: str = "auto"
    batch_size: int = 128
    micro_batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 3e-4
    cutoff_len: int = 256
    val_set_size: int = 2000
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["up_proj"])
    train_on_inputs: bool = True
    add_eos_token: bool = False
    group_by_length: bool = False
    resume_from_checkpoint: Optional[str] = None
    prompt_template_name: str = "alpaca"


class TokenizerHelper:
    def __init__(
        self, prompter, tokenizer, train_on_inputs, cutoff_len, add_eos_token=True
    ):
        self.prompter = prompter
        self.tokenizer = tokenizer
        self.train_on_inputs = train_on_inputs
        self.add_eos_token = add_eos_token
        self.cutoff_len = cutoff_len

    def tokenize(self, prompt):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            # Set padding to 'max_length' instead of False for GPTNeoXTokenizerFast???
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff_len
            and self.add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        return result

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)

        if not self.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = self.tokenize(user_prompt)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if self.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["input_ids"][
                user_prompt_len:
            ]  # could be sped up, probably
        else:
            tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"]
        # print(tokenized_full_prompt)
        return tokenized_full_prompt


def setup_model(config: TrainConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        device_map=config.device_map,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    return model, tokenizer


def load_data(tokenizer, config: TrainConfig) -> Tuple:
    """TODO: Not working yet.

    Args:
        config (TrainConfig): _description_

    Returns:
        Tuple: _description_
    """
    # Load the dataset
    dataset = load_dataset(config.dataset_name)

    tokenized_dataset = dataset.map(tokenizer, batched=True)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, mlm=False)

    # Split the dataset into train, validation and (optionally) test sets
    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"]

    if "test" in tokenized_dataset:
        test_dataset = tokenized_dataset["test"]
    else:
        test_dataset = None

    return train_dataset, val_dataset, test_dataset, data_collator


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # Prompt template to use, default to Alpaca
):
    if lora_target_modules is None:
        lora_target_modules = ["up_proj"]
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"\n\n\nLoRA fine-tuning model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    print(f"device map: {device_map}")

    #
    # Model loading
    #
    BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

    # for mpt
    config = AutoConfig.from_pretrained(
        "mistralai/Mistral-7B-v0.1", trust_remote_code=True, revision="main"
    )
    config.update({"max_seq_len": 4096})
    # config.attn_config['attn_impl'] = 'triton'

    model = AutoModelForCausalLM.from_pretrained(
        # 'mosaicml/mpt-7b',
        base_model,
        config=config,
        # base_model,
        # load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        # quantization_config=quantization_config,
        # load_in_8bit_fp32_cpu_offload=True
        trust_remote_code=True,
        revision="main",
    )

    # tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    #
    # 8-bit training
    #
    # had to turn int8 training off for some reason. could it be the titan rtx?
    # turned it on and kinda working now, but wtf?
    # model = prepare_model_for_int8_training(model)

    #
    # LoRA
    #
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved,
        # but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # Be more transparent about the % of trainable params.
    model.print_trainable_parameters()

    tokenizer_helper = TokenizerHelper(
        prompter, tokenizer, train_on_inputs, cutoff_len, add_eos_token
    )

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"]
            .shuffle()
            .map(tokenizer_helper.generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"]
            .shuffle()
            .map(tokenizer_helper.generate_and_tokenize_prompt)
        )
    else:
        train_data = (
            data["train"].shuffle().map(tokenizer_helper.generate_and_tokenize_prompt)
        )
        val_data = None

    print(train_data[0])

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism
        # when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    use_wandb = False
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            # fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name="wandb_run_name" if use_wandb else None,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    fire.Fire(train)
