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
from src.model    import load_model_and_tokenizer
from src.dataset  import load_and_prepare_dataset
from src.trainer  import train, save_model
from src.evaluate import run_evaluation


def main(cfg_path: str = "configs/config.yaml") -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["training"]["output_dir"], exist_ok=True)
    os.environ["WANDB_DISABLED"] = "true"

    logger.info("── 1/4  Loading model ───────────────────────")
    model, tokenizer = load_model_and_tokenizer(cfg)

    logger.info("── 2/4  Preparing dataset ───────────────────")
    dataset = load_and_prepare_dataset(cfg, tokenizer)

    logger.info("── 3/4  Training ────────────────────────────")
    trainer, history = train(model, tokenizer, dataset, cfg)
    save_model(trainer, tokenizer, cfg["training"]["output_dir"])

    logger.info("── 4/4  Evaluating ──────────────────────────")
    eval_out  = trainer.evaluate()
    eval_loss = eval_out.get("eval_loss", float("inf"))
    report    = run_evaluation(
        eval_loss,
        {"train": history.train_loss, "eval": history.eval_loss},
        cfg,
    )

    logger.info("\n" + "═" * 48)
    logger.info("  RESULTS")
    logger.info("═" * 48)
    logger.info(f"  Eval Loss  : {report['eval_loss']}")
    logger.info(f"  Perplexity : {report['perplexity']}")
    logger.info(f"  Outputs    : {cfg['training']['output_dir']}/")
    logger.info("═" * 48)


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    main(cfg_path)