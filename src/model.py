import logging
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(cfg: dict[str, Any]):
    """
    Load GPT-2 Medium tokenizer and model.
    No quantization needed — GPT-2 Medium (345M) fits on T4 in fp16.
    """
    model_name = cfg["model"]["name"]
    logger.info(f"Loading: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {total:,}")

    return model, tokenizer