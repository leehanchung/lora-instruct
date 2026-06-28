"""Training-monitoring callbacks for finetune.py.

`MonitorCallback` adds three signals on top of what `Trainer` already logs
(loss / grad_norm / lr / eval_loss):

  * **peak GPU memory** per logging step (catches creeping memory before an OOM),
  * **eval perplexity** (`exp(eval_loss)` — more interpretable than raw loss),
  * **sample generations** from a few fixed held-out prompts each eval (the
    qualitative "is it actually following instructions" signal).

Everything is rank-0 only and degrades gracefully without Weights & Biases: with
`--wandb_project` set it logs to W&B (aligned to the trainer's global step);
without it, the same numbers print to the console.
"""

import math

import torch
from transformers import TrainerCallback


def _wandb():
    """Return the active wandb module, or None if W&B isn't running."""
    try:
        import wandb

        if wandb.run is not None:
            return wandb
    except Exception:
        pass
    return None


class MonitorCallback(TrainerCallback):
    def __init__(
        self,
        prompter=None,
        tokenizer=None,
        sample_rows=None,
        sample_new_tokens=128,
        is_main=True,
    ):
        self.prompter = prompter
        self.tokenizer = tokenizer
        self.sample_rows = list(sample_rows or [])
        self.sample_new_tokens = sample_new_tokens
        self.is_main = is_main
        self._history = []  # (step, instruction, completion) across evals

    # ---- peak GPU memory ----
    def on_train_begin(self, args, state, control, **kwargs):
        if self.is_main and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not (self.is_main and torch.cuda.is_available()):
            return
        dev = torch.cuda.current_device()
        gb = round(torch.cuda.max_memory_allocated(dev) / 1e9, 2)
        torch.cuda.reset_peak_memory_stats(dev)
        print(f"  [monitor] gpu{dev}_peak_gb={gb}", flush=True)
        wb = _wandb()
        if wb:
            wb.log({f"system/gpu{dev}_peak_gb": gb}, step=state.global_step)

    # ---- eval perplexity + sample generations ----
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.is_main:
            return
        wb = _wandb()
        if metrics and "eval_loss" in metrics:
            try:
                ppl = math.exp(metrics["eval_loss"])
            except OverflowError:
                ppl = float("inf")
            print(
                f"  [monitor] eval_loss={metrics['eval_loss']:.4f} perplexity={ppl:.2f}",
                flush=True,
            )
            if wb:
                wb.log({"eval/perplexity": ppl}, step=state.global_step)
        if self.sample_rows and self.tokenizer is not None:
            self._log_samples(state, kwargs.get("model"), wb)

    def _log_samples(self, state, model, wb):
        if model is None:
            return
        gen_model = getattr(model, "module", model)  # unwrap DDP
        was_training = gen_model.training
        prev_cache = gen_model.config.use_cache
        gen_model.eval()
        gen_model.config.use_cache = True  # generation is much faster with the kv-cache
        device = next(gen_model.parameters()).device
        try:
            for r in self.sample_rows:
                instruction = r.get("instruction", "")
                prompt = (
                    self.prompter.generate_prompt(instruction, r.get("input", ""))
                    if self.prompter
                    else instruction
                )
                ids = self.tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = gen_model.generate(
                        **ids, max_new_tokens=self.sample_new_tokens, do_sample=False
                    )
                completion = self.tokenizer.decode(
                    out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
                )
                self._history.append((state.global_step, instruction, completion))
        finally:
            gen_model.config.use_cache = prev_cache
            if was_training:
                gen_model.train()

        if self._history:
            _, instr, comp = self._history[-1]
            print(
                f"  [monitor] sample @step {state.global_step}: "
                f"{instr[:60]!r} -> {comp[:140]!r}",
                flush=True,
            )
        if wb:
            table = wb.Table(columns=["step", "instruction", "completion"])
            for step, instr, comp in self._history:
                table.add_data(step, instr, comp)
            wb.log({"samples": table}, step=state.global_step)
