# Fine-tune Gemma 2B on Domain-Specific Data with QLoRA

Parameter-efficient fine-tuning of an open-source LLM using 4-bit quantization and LoRA adapters — runnable on a free Colab T4 GPU.

---

## Overview

This project demonstrates how to adapt a large language model to a custom domain without full fine-tuning. Instead of updating 2 billion parameters, QLoRA freezes the quantized base model and trains only a small set of low-rank adapter matrices — roughly 1% of total parameters — while achieving competitive performance.

| | |
|---|---|
| **Base model** | `google/gemma-2b` |
| **Method** | QLoRA (4-bit NF4) + LoRA adapters |
| **Trainable params** | ~1% of total |
| **Hardware** | Google Colab Free (T4 16GB) |
| **Framework** | PyTorch · HuggingFace · PEFT · TRL |

---

## Results

| Metric | Value |
|--------|-------|
| Perplexity | — |
| ROUGE-1 | — |
| ROUGE-2 | — |
| ROUGE-L | — |

> Fill in after training. Charts saved to `outputs/`.

**Loss curves**

![Loss curves](outputs/loss_curves.png)

**ROUGE scores**

![ROUGE scores](outputs/rouge_scores.png)

---

## Project Structure

```
llm-finetune/
├── configs/
│   └── config.yaml        # all hyperparameters and LoRA settings
├── src/
│   ├── model.py           # QLoRA model loading + LoRA adapter setup
│   ├── dataset.py         # dataset loading, prompt template, tokenization
│   ├── trainer.py         # SFTTrainer + loss recording callback
│   ├── evaluate.py        # perplexity, ROUGE, matplotlib plots
│   └── inference.py       # load adapter and generate responses
├── train.py               # entry point — runs the full pipeline
├── notebook.ipynb         # Colab-ready notebook with inline plots
└── requirements.txt
```

---

## Quickstart

### Option A — Colab (recommended)

1. Open `notebook.ipynb` in Google Colab
2. Set runtime to **T4 GPU**
3. Add your HuggingFace token to Colab secrets as `HF_TOKEN`
4. Update `DATASET_NAME` in cell 3
5. Run all cells

### Option B — Local

```bash
git clone https://github.com/YOUR_USERNAME/llm-finetune
cd llm-finetune

pip install -r requirements.txt

# Set your HuggingFace token
export HF_TOKEN=your_token_here

# Update dataset name in config
# configs/config.yaml → data.dataset_name

python train.py
```

---

## How It Works

### QLoRA — 4-bit quantization

The base model is loaded in 4-bit NF4 precision using `bitsandbytes`. This reduces Gemma 2B's memory footprint from ~16GB (fp16) to ~5GB, making it fit on a free T4 GPU. The quantized weights are frozen — no gradients flow through them.

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

### LoRA — low-rank adaptation

Instead of updating the full weight matrices, LoRA injects two small trainable matrices `A` and `B` into the attention layers. The weight update becomes `ΔW = A × B` where rank `r=16` — far fewer parameters than the original matrix.

```
Target layers: q_proj, k_proj, v_proj, o_proj
Rank (r):      16
Alpha:         32
Trainable:     ~1% of total params
```

### Training

Supervised fine-tuning with `SFTTrainer` from TRL. A custom callback records train and eval loss at every step for plotting. Gradient checkpointing is enabled to trade compute for memory.

### Evaluation

- **Perplexity** — computed from eval loss. Measures how well the model predicts the validation tokens.
- **ROUGE-1 / 2 / L** — computed by generating responses on 100 validation samples and comparing against ground truth.

---

## Configuration

All settings live in `configs/config.yaml`. The only lines you need to change:

```yaml
model:
  name: "google/gemma-2b"      # swap for any HF causal LM

data:
  dataset_name: "REPLACE_ME"   # your HuggingFace dataset
```

---

## Outputs

After training, `outputs/` contains:

```
outputs/
├── lora_adapter/          # LoRA weights — load with PeftModel
├── loss_curves.png        # train + eval loss over steps
├── rouge_scores.png       # ROUGE bar chart
├── eval_report.json       # perplexity + ROUGE as JSON
└── metrics_history.json   # raw step-by-step loss logs
```

---

## Inference

```python
from src.inference import load_finetuned, generate
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

model, tokenizer = load_finetuned(cfg, "outputs/lora_adapter")

response = generate(
    model, tokenizer,
    instruction="Your instruction here",
    input_text="Optional context",
    cfg=cfg,
)
print(response)
```

Or via CLI:

```bash
python src/inference.py configs/config.yaml outputs/lora_adapter
```

---

## Requirements

- Python 3.10+
- CUDA GPU (T4 16GB minimum)
- HuggingFace account with Gemma access

---

## References

- [QLoRA paper](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [LoRA paper](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [Gemma model](https://huggingface.co/google/gemma-2b) — Google, 2024
- [PEFT library](https://github.com/huggingface/peft)
- [TRL library](https://github.com/huggingface/trl)