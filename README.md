# Fine-tune GPT-2 Medium on Sci-Fi Literature with QLoRA

A production-structured NLP project demonstrating domain-specific language model fine-tuning using QLoRA. GPT-2 Medium (345M) is adapted from general English text to generate coherent, stylistically authentic science fiction prose using classic works from Project Gutenberg.

---

## The Problem This Solves

General-purpose language models produce generic output that lacks the narrative style, vocabulary, and thematic depth of science fiction. Fine-tuning on a curated domain corpus forces the model to specialize.

---

## Why This Approach

| Approach | Problem |
|---|---|
| Train from scratch | Needs billions of tokens, weeks of compute |
| Prompt engineering | No weight updates, model doesn't learn the domain |
| Full fine-tuning on 7B+ model | Too large for free GPU |
| **This project** | **GPT-2 Medium (345M) + QLoRA on free T4 — fast, safe, proven** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        config.yaml                                  │
│            (all hyperparameters in one place)                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. MODEL LOADING  (src/model.py)                                   │
│  ┌─────────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  GPT-2 Medium   │───▶│ BitsAndBytes     │───▶│  PEFT / LoRA  │  │
│  │  (345M params)  │    │ 4-bit NF4        │    │  Adapters     │  │
│  │  HuggingFace    │    │ Quantization     │    │  (c_attn,     │  │
│  │                 │    │                  │    │   c_proj)     │  │
│  └─────────────────┘    └──────────────────┘    └───────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. DATASET  (src/Dataset.py)                                       │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │ CSV Load │──▶│ Cleaning │──▶│  Split   │──▶│  Tokenization  │  │
│  │ (pandas) │   │ - nulls  │   │ train/   │   │  pad to        │  │
│  │          │   │ - short  │   │ val      │   │  max_length    │  │
│  │          │   │ - dupes  │   │          │   │                │  │
│  └──────────┘   └──────────┘   └──────────┘   └────────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. TRAINING  (src/trainer.py)                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              HuggingFace Trainer                               │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐ │  │
│  │  │  Training-  │  │  DataCollator│  │  MetricsCallback     │ │  │
│  │  │  Arguments  │  │  (LM, no MLM)│  │  (loss, lr, grad)    │ │  │
│  │  └─────────────┘  └──────────────┘  └──────────────────────┘ │  │
│  │                                                               │  │
│  │  Optimizer: paged_adamw_8bit                                  │  │
│  │  Scheduler: cosine with warmup                                │  │
│  │  Mixed precision: bf16/fp16                                   │  │
│  └───────────────────────────┬───────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Save LoRA adapters + tokenizer ──▶ outputs/                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. EVALUATION  (src/evaluate.py)                                   │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Perplexity      │  │  Loss Curves │  │  eval_report.json    │  │
│  │  exp(eval_loss)  │  │  (matplotlib)│  │  (loss, perplexity)  │  │
│  └──────────────────┘  └──────────────┘  └──────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. INFERENCE  (src/inference.py)                                   │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Load saved      │  │  HuggingFace │  │  Text Generation     │  │
│  │  LoRA model      │──▶│  pipeline    │──▶  (temperature, top_p,│  │
│  │  + tokenizer     │  │              │  │   repetition_pen)    │  │
│  └──────────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline Data Flow

| Stage | Module | Input | Output |
|-------|--------|-------|--------|
| **1. Config** | `configs/config.yaml` | User edits | `cfg` dict |
| **2. Model** | `src/model.py` | `cfg` | 4-bit GPT-2 + LoRA adapters |
| **3. Dataset** | `src/Dataset.py` | `cfg` + `tokenizer` | `DatasetDict` (train/test) |
| **4. Train** | `src/trainer.py` | model + dataset + cfg | Trained model + metrics |
| **5. Eval** | `src/evaluate.py` | eval_loss + history | perplexity + loss plots |
| **6. Inference** | `src/inference.py` | saved model + prompt | Generated text |

---

## Model & Dataset

| | |
|---|---|
| **Base model** | `gpt2-medium` — 345M parameters |
| **Fine-tuning** | QLoRA (4-bit NF4 quantization + LoRA on `c_attn`, `c_proj`) |
| **Dataset** | [Sci-Fi Books — Project Gutenberg](https://huggingface.co/datasets/stevez80/Sci-Fi-Books-gutenberg) |
| **Hardware** | Google Colab Free Tier — NVIDIA T4 (16GB VRAM) |
| **Training time** | ~25 minutes |
| **Framework** | PyTorch + HuggingFace Transformers + PEFT |

---

## Project Structure

```
LLM-Fine-Tuning-Pipeline/
├── configs/
│   └── config.yaml              # all hyperparameters
├── src/
│   ├── model.py                 # model + tokenizer loading (4-bit QLoRA)
│   ├── Dataset.py               # CSV loading, cleaning, tokenization
│   ├── trainer.py               # Trainer + metrics callback
│   ├── evaluate.py              # perplexity + loss curves
│   └── inference.py             # load saved model + generate text
├── train.py                     # entry point — runs full pipeline
├── GPT2-QLoRA-SciFi.ipynb      # Colab notebook (main focus)
├── requirements.txt
└── README.md
```

---

## Metrics Tracked

| Metric | Description |
|---|---|
| `eval_loss` | Cross-entropy loss on validation set |
| `perplexity` | exp(eval_loss) — lower = better |
| `token_accuracy` | % of correctly predicted tokens |
| `train_loss` | Per-step training loss |
| `learning_rate` | LR schedule over training |
| `grad_norm` | Gradient norm (detects instability) |

---

## Quickstart

### Option A — Colab (recommended)

1. Open `GPT2-QLoRA-SciFi.ipynb` in Google Colab
2. Set runtime to **T4 GPU** — Runtime → Change runtime type → T4
3. Run all cells top to bottom
4. Training completes in ~25 minutes

### Option B — Local

```bash
git clone https://github.com/aieng-abdullah/LLM-Fine-Tuning-Pipeline
cd LLM-Fine-Tuning-Pipeline
pip install -r requirements.txt
python train.py
```

### Inference only

```python
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model     = AutoModelForCausalLM.from_pretrained("./finetuned-gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("./finetuned-gpt2-medium")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

out = generator(
    "In a distant galaxy, a lone spacecraft",
    max_new_tokens=200,
    temperature=0.8,
    do_sample=True,
)
print(out[0]["generated_text"])
```

---

## Key Engineering Decisions

**Why QLoRA on GPT-2 Medium?**
GPT-2 Medium at 345M fits in fp16 on T4, but QLoRA (4-bit) reduces memory to ~3-4 GiB, leaving headroom for larger batches and faster training.

**Why these LoRA target modules?**
GPT-2 uses Conv1D layers (`c_attn`, `c_proj`) instead of separate Q/K/V projections. These are the attention layers — LoRA adapters here capture the domain-specific attention patterns.

**Why gradient checkpointing?**
Trades compute for memory by recomputing activations during backward pass. Essential for fitting model + LoRA + optimizer in T4 VRAM.

**Why paged AdamW?**
Memory-efficient optimizer that offloads optimizer states to CPU when VRAM is tight. Standard for QLoRA training.

---

## Configuration

All hyperparameters live in `configs/config.yaml`. To experiment:

```yaml
model:
  name: "gpt2-medium"

qlora:
  lora_r: 32          # increase for more capacity
  lora_alpha: 64      # usually 2x lora_r
  lora_dropout: 0.1   # increase for regularization

training:
  num_train_epochs: 3
  learning_rate: 1e-4
```
