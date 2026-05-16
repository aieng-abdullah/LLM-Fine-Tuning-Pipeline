import json
import logging
import math
import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logger = logging.getLogger(__name__)


def compute_perplexity(eval_loss: float) -> float:
    return round(math.exp(eval_loss), 4)


def plot_loss_curves(history: dict, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    if history["train"]:
        ax.plot(
            [p["step"] for p in history["train"]],
            [p["loss"] for p in history["train"]],
            color="#5C6BC0", lw=1.8, label="Train loss",
        )
    if history["eval"]:
        ax.plot(
            [p["step"] for p in history["eval"]],
            [p["loss"] for p in history["eval"]],
            color="#EF5350", lw=1.8, ls="--", label="Eval loss",
        )

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training & Evaluation Loss — GPT-2 Medium", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = os.path.join(output_dir, "loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Loss curve → {path}")
    return path


def run_evaluation(eval_loss: float, history: dict, cfg: dict[str, Any]) -> dict:
    output_dir = cfg["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    perplexity = compute_perplexity(eval_loss)
    logger.info(f"Perplexity: {perplexity}")

    plot_loss_curves(history, output_dir)

    report = {"eval_loss": round(eval_loss, 4), "perplexity": perplexity}
    path = os.path.join(output_dir, "eval_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report → {path}")

    return report