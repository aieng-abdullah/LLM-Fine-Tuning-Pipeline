import logging
from typing import Any

from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def build_prompt(example: dict, template: str) -> dict:
    """Fill prompt template with example fields."""
    return {
        "text": template.format(
            instruction=example.get("instruction", ""),
            input=example.get("input", ""),
            output=example.get("output", ""),
        )
    }


def tokenize(batch: dict, tokenizer: PreTrainedTokenizer, max_len: int) -> dict:
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_len,
        padding=False,         # DataCollator handles padding at batch time
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def load_and_prepare_dataset(
    cfg: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
) -> DatasetDict:
    """
    Steps
    -----
    1. Load raw dataset from HuggingFace hub (or local path)
    2. Train / val split
    3. Apply prompt template
    4. Tokenize — labels == input_ids (causal LM)
    5. Return DatasetDict with 'train' and 'test' keys
    """
    data_cfg = cfg["data"]
    template = data_cfg["prompt_template"]
    max_len  = data_cfg["max_seq_length"]

    logger.info(f"Loading dataset: {data_cfg['dataset_name']}")
    raw = load_dataset(
        data_cfg["dataset_name"],
        split=data_cfg["dataset_split"],
    )

    # ── Train / val split ─────────────────────────────────────────────────────
    split = raw.train_test_split(
        test_size=data_cfg["val_split_ratio"],
        seed=cfg["training"]["seed"],
    )
    logger.info(f"Split — train: {len(split['train']):,}  val: {len(split['test']):,}")

    # ── Prompt template ───────────────────────────────────────────────────────
    split = split.map(
        lambda ex: build_prompt(ex, template),
        remove_columns=split["train"].column_names,
        desc="Applying prompt template",
    )

    # ── Tokenize ──────────────────────────────────────────────────────────────
    split = split.map(
        lambda batch: tokenize(batch, tokenizer, max_len),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    split.set_format("torch")

    # ── Sanity check ──────────────────────────────────────────────────────────
    sample_len = split["train"][0]["input_ids"].shape[0]
    logger.info(f"Sample token length: {sample_len} / {max_len}")

    return split