import json
import logging
import math
import os
from typing import Any

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datasets import Dataset
from evaluate import load as load_metric
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


# ── Perplexity ────────────────────────────────────────────────────────────────

def compute_perplexity(eval_loss: float) -> float:
    """Lower is better. A perfect model → 1.0, random model → vocab size."""
    return round(math.exp(eval_loss), 4)


# ── ROUGE ─────────────────────────────────────────────────────────────────────

def compute_rouge(
    model,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    cfg: dict[str, Any],
    n_samples: int = 100,
) -> dict[str, float]:
    """
    Generate responses for n_samples from the val set,
    then score against ground-truth with ROUGE.
    """
    rouge  = load_metric("rouge")
    device = next(model.parameters()).device
    inf    = cfg["inference"]

    preds, refs = [], []
    model.eval()

    with torch.no_grad():
        for ex in list(dataset)[:n_samples]:
            full = tokenizer.decode(ex["input_ids"], skip_special_tokens=True)

            marker = "### Response:"
            if marker in full:
                prompt = full[: full.index(marker) + len(marker)]
                ref    = full[full.index(marker) + len(marker):].strip()
            else:
                prompt, ref = full, ""

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=inf["max_new_tokens"],
                temperature=inf["temperature"],
                top_p=inf["top_p"],
                repetition_penalty=inf["repetition_penalty"],
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            pred = tokenizer.decode(
                out[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            )
            preds.append(pred.strip())
            refs.append(ref)

    scores = rouge.compute(predictions=preds, references=refs)
    return {k: round(v, 4) for k, v in scores.items()}


# ── Plots ─────────────────────────────────────────────────────────────────────

def _base_ax(figsize=(9, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def plot_loss_curves(history: dict, output_dir: str) -> str:
    fig, ax = _base_ax()

    if history["train"]:
        steps  = [p["step"] for p in history["train"]]
        losses = [p["loss"] for p in history["train"]]
        ax.plot(steps, losses, color="#5C6BC0", lw=1.8, label="Train loss")

    if history["eval"]:
        steps  = [p["step"] for p in history["eval"]]
        losses = [p["loss"] for p in history["eval"]]
        ax.plot(steps, losses, color="#EF5350", lw=1.8, ls="--", label="Eval loss")

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training & Evaluation Loss", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.legend(fontsize=10)
    fig.tight_layout()

    path = os.path.join(output_dir, "loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Loss curve → {path}")
    return path


def plot_rouge(scores: dict[str, float], output_dir: str) -> str:
    keys   = [k for k in scores if k.startswith("rouge")]
    vals   = [scores[k] for k in keys]
    colors = ["#5C6BC0", "#26A69A", "#EF5350", "#FFA726"]

    fig, ax = _base_ax(figsize=(6, 4))
    bars = ax.bar(keys, vals, color=colors[:len(keys)], width=0.5, edgecolor="white")

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylim(0, min(1.0, max(vals) * 1.3))
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("ROUGE Scores — validation set", fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(output_dir, "rouge_scores.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROUGE chart → {path}")
    return path


# ── Full eval pipeline ────────────────────────────────────────────────────────

def run_evaluation(
    model,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Dataset,
    eval_loss: float,
    history: dict,
    cfg: dict[str, Any],
) -> dict:
    output_dir = cfg["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    perplexity   = compute_perplexity(eval_loss)
    rouge_scores = compute_rouge(model, tokenizer, eval_dataset, cfg)

    logger.info(f"Perplexity : {perplexity}")
    logger.info(f"ROUGE      : {rouge_scores}")

    plot_loss_curves(history, output_dir)
    plot_rouge(rouge_scores, output_dir)

    report = {"perplexity": perplexity, **rouge_scores}
    path   = os.path.join(output_dir, "eval_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report     → {path}")

    return report