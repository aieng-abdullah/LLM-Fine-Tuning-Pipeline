import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


def _bnb_config(cfg: dict[str, Any]) -> BitsAndBytesConfig:
    m = cfg["model"]
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    return BitsAndBytesConfig(
        load_in_4bit=m["load_in_4bit"],
        bnb_4bit_quant_type=m["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=dtype_map.get(m["bnb_4bit_compute_dtype"], torch.float16),
        bnb_4bit_use_double_quant=m["bnb_4bit_use_double_quant"],
    )


def _lora_config(cfg: dict[str, Any]) -> LoraConfig:
    l = cfg["lora"]
    return LoraConfig(
        r=l["r"],
        lora_alpha=l["lora_alpha"],
        target_modules=l["target_modules"],
        lora_dropout=l["lora_dropout"],
        bias=l["bias"],
        task_type=l["task_type"],
    )


def load_model_and_tokenizer(cfg: dict[str, Any]):
    """
    1. Load Gemma tokenizer
    2. Load base model in 4-bit (QLoRA)
    3. Prepare for k-bit training (freeze quantized layers)
    4. Attach LoRA adapters
    5. Return model, tokenizer, param stats
    """
    model_name = cfg["model"]["name"]
    logger.info(f"Loading: {model_name}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        add_eos_token=True,      # Gemma needs this for clean generation stops
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Base model in 4-bit ───────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=_bnb_config(cfg),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # ── Prepare for QLoRA training ────────────────────────────────────────────
    # This step freezes the 4-bit weights so only LoRA adapters are updated
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,   # saves VRAM on Colab free T4
    )

    # ── Attach LoRA adapters ──────────────────────────────────────────────────
    model = get_peft_model(model, _lora_config(cfg))

    # ── Parameter summary ─────────────────────────────────────────────────────
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stats = {
        "total_params":     total,
        "trainable_params": trainable,
        "trainable_pct":    round(100 * trainable / total, 3),
    }
    logger.info(
        f"Trainable: {trainable:,} / {total:,} ({stats['trainable_pct']}%)"
    )
    return model, tokenizer, stats