import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logger = logging.getLogger(__name__)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_finetuned(cfg: dict[str, Any], adapter_path: str):
    """
    Load base Gemma 2B in 4-bit, then merge the LoRA adapter on top.

    Parameters
    ----------
    cfg          : full config dict
    adapter_path : path to saved lora_adapter/ folder

    Returns
    -------
    model, tokenizer
    """
    m = cfg["model"]
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}

    bnb = BitsAndBytesConfig(
        load_in_4bit=m["load_in_4bit"],
        bnb_4bit_quant_type=m["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=dtype_map.get(m["bnb_4bit_compute_dtype"], torch.float16),
        bnb_4bit_use_double_quant=m["bnb_4bit_use_double_quant"],
    )

    # Load tokenizer from adapter path — saved alongside weights in trainer.py
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        m["name"],
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    logger.info(f"Loaded fine-tuned model from {adapter_path}")
    return model, tokenizer


# ── Generate ──────────────────────────────────────────────────────────────────

def generate(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    cfg: dict[str, Any] | None = None,
) -> str:
    """
    Build a prompt from instruction + input, generate and return response text.

    Parameters
    ----------
    instruction : task description
    input_text  : optional context (leave empty if not needed)
    cfg         : config dict — uses inference section for generation params
    """
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )

    inf    = cfg["inference"] if cfg else {}
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=inf.get("max_new_tokens", 256),
            temperature=inf.get("temperature", 0.7),
            top_p=inf.get("top_p", 0.9),
            repetition_penalty=inf.get("repetition_penalty", 1.1),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens — not the prompt
    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    return response.strip()


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import yaml

    cfg_path     = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    adapter_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/lora_adapter"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model, tokenizer = load_finetuned(cfg, adapter_path)

    print("\n── Fine-tuned Gemma 2B — interactive demo ──\n")
    print("Type 'quit' to exit.\n")

    while True:
        instruction = input("Instruction: ").strip()
        if instruction.lower() == "quit":
            break
        input_text = input("Input (optional, press Enter to skip): ").strip()
        response   = generate(model, tokenizer, instruction, input_text, cfg)
        print(f"\nResponse:\n{response}\n{'-'*50}\n")