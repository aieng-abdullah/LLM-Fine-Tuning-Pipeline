import logging
from typing import Any

from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_dataset(raw):
    """
    MedQuAD-specific cleaning:
    - Drop rows where answer is null (31k records had answers removed for copyright)
    - Drop rows where answer is too short to be meaningful
    - Strip whitespace from question and answer
    """
    before = len(raw)

    raw = raw.filter(
        lambda x: x["answer"] is not None and len(x["answer"].strip()) > 20,
        desc="Filtering null/short answers",
    )

    raw = raw.map(
        lambda x: {
            "question": x["question"].strip() if x["question"] else "",
            "answer":   x["answer"].strip()   if x["answer"]   else "",
            "question_focus": x["question_focus"].strip() if x["question_focus"] else "",
        },
        desc="Stripping whitespace",
    )

    after = len(raw)
    logger.info(f"Cleaning: {before:,} → {after:,} rows ({before - after:,} dropped)")
    return raw


# ── Prompt formatting ─────────────────────────────────────────────────────────

def build_prompt(example: dict, template: str) -> dict:
    """
    MedQuAD fields:
      question       → instruction
      question_focus → input  (disease/topic name — gives the model context)
      answer         → output
    """
    return {
        "text": template.format(
            instruction=example.get("question", ""),
            input=example.get("question_focus", ""),
            output=example.get("answer", ""),
        )
    }


# ── Tokenize ──────────────────────────────────────────────────────────────────

def tokenize(batch: dict, tokenizer: PreTrainedTokenizer, max_len: int) -> dict:
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


# ── Main loader ───────────────────────────────────────────────────────────────

def load_and_prepare_dataset(
    cfg: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
) -> DatasetDict:
    """
    1. Load MedQuAD from HuggingFace Hub
    2. Clean — drop null/short answers
    3. Train / val split
    4. Apply prompt template
    5. Tokenize
    6. Return DatasetDict with 'train' and 'test' keys
    """
    data_cfg = cfg["data"]
    template = data_cfg["prompt_template"]
    max_len  = data_cfg["max_seq_length"]

    logger.info(f"Loading dataset: {data_cfg['dataset_name']}")
    raw = load_dataset(
        data_cfg["dataset_name"],
        split=data_cfg["dataset_split"],
    )

    # ── Clean ─────────────────────────────────────────────────────────────────
    raw = clean_dataset(raw)

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

    sample_len = split["train"][0]["input_ids"].shape[0]
    logger.info(f"Sample token length: {sample_len} / {max_len}")

    return split