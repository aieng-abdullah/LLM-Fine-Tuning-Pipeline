import logging
import os
import sys

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

sys.path.insert(0, os.path.dirname(__file__))
from src.model   import load_model_and_tokenizer
from src.dataset import load_and_prepare_dataset
from src.trainer import train, save_adapter
from src.evaluate import run_evaluation


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {path}")
    return cfg


def main(cfg_path: str = "configs/config.yaml") -> None:
    cfg = load_config(cfg_path)
    os.makedirs(cfg["training"]["output_dir"], exist_ok=True)

    # ── Step 1: Model ─────────────────────────────────────────────────────────
    logger.info("── 1/4  Loading model ───────────────────────")
    model, tokenizer, param_stats = load_model_and_tokenizer(cfg)
    logger.info(
        f"    Trainable params: {param_stats['trainable_params']:,} "
        f"({param_stats['trainable_pct']}%)"
    )

    # ── Step 2: Dataset ───────────────────────────────────────────────────────
    logger.info("── 2/4  Preparing dataset ───────────────────")
    dataset = load_and_prepare_dataset(cfg, tokenizer)

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    logger.info("── 3/4  Training ────────────────────────────")
    trainer, history = train(model, tokenizer, dataset, cfg)
    save_adapter(trainer, cfg["training"]["output_dir"])

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    logger.info("── 4/4  Evaluating ──────────────────────────")
    eval_out  = trainer.evaluate()
    eval_loss = eval_out.get("eval_loss", float("inf"))

    report = run_evaluation(
        model=trainer.model,
        tokenizer=tokenizer,
        eval_dataset=dataset["test"],
        eval_loss=eval_loss,
        history={"train": history.train_loss, "eval": history.eval_loss},
        cfg=cfg,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("═" * 48)
    logger.info("  RESULTS")
    logger.info("═" * 48)
    logger.info(f"  Perplexity  : {report['perplexity']}")
    for k, v in report.items():
        if k.startswith("rouge"):
            logger.info(f"  {k.upper():<12}: {v:.4f}")
    logger.info(f"  Outputs     : {cfg['training']['output_dir']}/")
    logger.info("═" * 48)


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    main(cfg_path)