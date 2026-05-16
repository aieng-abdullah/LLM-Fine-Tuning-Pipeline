import logging
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_and_prepare_dataset(
    cfg: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
) -> DatasetDict:
    """
    1. Load CSV from local path
    2. Clean — drop empty/short text rows
    3. Train / val split
    4. Tokenize with padding to max_length
    5. Return DatasetDict with 'train' and 'test' keys
    """
    data_cfg = cfg["data"]
    max_len   = data_cfg["max_seq_length"]
    col       = data_cfg["text_column"]

    logger.info(f"Loading CSV: {data_cfg['csv_path']}")
    df = pd.read_csv(data_cfg["csv_path"])

    # ── Clean ─────────────────────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=[col])
    df = df[df[col].str.strip().str.len() > 50]
    df = df.reset_index(drop=True)
    logger.info(f"Cleaned: {before:,} → {len(df):,} rows")

    # ── HuggingFace Dataset ───────────────────────────────────────────────────
    raw   = Dataset.from_pandas(df)
    split = raw.train_test_split(
        test_size=data_cfg["val_split_ratio"],
        seed=cfg["training"]["seed"],
    )
    logger.info(f"Split — train: {len(split['train']):,}  val: {len(split['test']):,}")

    # ── Tokenize ──────────────────────────────────────────────────────────────
    remove_cols = [c for c in split["train"].column_names if c != col]

    def tokenize(batch):
        tokens = tokenizer(
            batch[col],
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    split = split.map(
        tokenize,
        batched=True,
        remove_columns=split["train"].column_names,
        desc="Tokenizing",
    )
    split.set_format("torch")

    logger.info(f"Token shape: {split['train'][0]['input_ids'].shape}")
    return split