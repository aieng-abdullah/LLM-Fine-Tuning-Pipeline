import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
)
from datasets import DatasetDict

logger = logging.getLogger(__name__)


# ── Metric history ────────────────────────────────────────────────────────────

@dataclass
class MetricsHistory:
    train_loss: list = field(default_factory=list)
    eval_loss:  list = field(default_factory=list)

    def save(self, output_dir: str) -> None:
        path = os.path.join(output_dir, "metrics_history.json")
        with open(path, "w") as f:
            json.dump({"train": self.train_loss, "eval": self.eval_loss}, f, indent=2)
        logger.info(f"Metrics saved → {path}")


# ── Callback ──────────────────────────────────────────────────────────────────

class LossRecorderCallback(TrainerCallback):
    def __init__(self, history: MetricsHistory):
        self.history = history

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kw):
        if not logs:
            return
        step = state.global_step
        if "loss" in logs:
            self.history.train_loss.append({"step": step, "loss": round(logs["loss"], 4)})
        if "eval_loss" in logs:
            self.history.eval_loss.append({"step": step, "loss": round(logs["eval_loss"], 4)})


# ── Training args ─────────────────────────────────────────────────────────────

def _build_args(cfg: dict[str, Any]) -> TrainingArguments:
    t = cfg["training"]
    return TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"],
        fp16=t["fp16"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        report_to=t["report_to"],
        seed=t["seed"],
        dataloader_pin_memory=False,
    )


# ── Train ─────────────────────────────────────────────────────────────────────

def train(model, tokenizer, dataset: DatasetDict, cfg: dict[str, Any]):
    history  = MetricsHistory()
    args     = _build_args(cfg)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        callbacks=[LossRecorderCallback(history)],
    )

    logger.info("Training started ...")
    trainer.train()
    logger.info("Training complete.")

    os.makedirs(cfg["training"]["output_dir"], exist_ok=True)
    history.save(cfg["training"]["output_dir"])

    return trainer, history


# ── Save ──────────────────────────────────────────────────────────────────────

def save_model(trainer: Trainer, tokenizer, output_dir: str) -> None:
    path = os.path.join(output_dir, "finetuned-gpt2-medium")
    trainer.model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    logger.info(f"Model saved → {path}")