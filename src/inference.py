import logging
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


def load_finetuned(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForCausalLM.from_pretrained(model_path)
    logger.info(f"Loaded model from {model_path}")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, cfg: dict[str, Any] | None = None) -> str:
    inf = cfg["inference"] if cfg else {}
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    out = generator(
        prompt,
        max_new_tokens=inf.get("max_new_tokens", 200),
        temperature=inf.get("temperature", 0.8),
        top_p=inf.get("top_p", 0.9),
        repetition_penalty=inf.get("repetition_penalty", 1.1),
        do_sample=True,
        truncation=True,
    )
    return out[0]["generated_text"]


if __name__ == "__main__":
    import sys, yaml

    cfg_path   = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/finetuned-gpt2-medium"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model, tokenizer = load_finetuned(model_path)

    print("\n── GPT-2 Medium Sci-Fi Generator ──\n")
    while True:
        prompt = input("Prompt (or 'quit'): ").strip()
        if prompt.lower() == "quit":
            break
        print("\n" + generate(model, tokenizer, prompt, cfg) + "\n" + "-"*50 + "\n")