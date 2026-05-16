# Fine-tune GPT-2 Medium on Sci-Fi Literature

A production-structured NLP project demonstrating domain-specific language model fine-tuning. GPT-2 Medium is adapted from general English text to generate coherent, stylistically authentic science fiction prose using classic works from Project Gutenberg.

---

## The Problem This Solves

General-purpose language models like GPT-2 are trained on broad internet text. When asked to generate science fiction, they produce generic output that lacks the narrative style, vocabulary, and thematic depth of the genre.

**The core problem:** A model trained on everything knows a little about everything but masters nothing.

**The solution:** Fine-tuning on a curated domain corpus forces the model to specialize. After training on Sci-Fi literature, the model generates text that reflects the genre's distinctive patterns — interstellar settings, technical language, character archetypes, and narrative tension — rather than generic prose.

This is the same principle behind production LLMs used in legal document generation, medical record summarization, and code completion: take a strong base model and adapt it cheaply to a specific domain.

---

## Why This Approach

| Approach | Problem |
|---|---|
| Train from scratch | Needs billions of tokens, weeks of compute, not feasible |
| Prompt engineering | No weight updates, model doesn't actually learn the domain |
| Full fine-tuning on 7B+ model | Too large for free GPU, requires QLoRA/quantization |
| **This project** | **GPT-2 Medium (345M) is small enough to fully fine-tune on a free T4 in ~14 minutes** |

---

## Model & Dataset

| | |
|---|---|
| **Base model** | `gpt2-medium` — 345M parameters, 24 layers, 16 attention heads |
| **Dataset** | [Sci-Fi Books — Project Gutenberg](https://huggingface.co/datasets/stevez80/Sci-Fi-Books-gutenberg) |
| **Domain** | Classic science fiction — H.G. Wells, Jules Verne, Edgar Rice Burroughs and others |
| **Hardware** | Google Colab Free Tier — NVIDIA T4 (16GB VRAM) |
| **Training time** | ~14 minutes for 2 epochs |
| **Framework** | PyTorch · HuggingFace Transformers |

---

## Results

| Metric | Value |
|--------|-------|
| Train loss | 0.456 |
| Total steps | 5,366 |
| Epochs | 2 |
| Train samples/sec | 6.576 |

**Loss curves — model converging over training**

![Loss curves](outputs/loss_curves.png)

**Sample generation**

```
Prompt     : In a distant galaxy, a lady, the daughter of a prince,
             is called to the palace in an attempt to bring her husband
             back to civilization.
             Written by David P. Williams
```

---

## Project Structure

```
gpt2-finetune/
├── configs/
│   └── config.yaml        # all hyperparameters in one place
├── src/
│   ├── model.py           # model + tokenizer loading
│   ├── dataset.py         # CSV loading, cleaning, tokenization
│   ├── trainer.py         # HuggingFace Trainer + loss recording callback
│   ├── evaluate.py        # perplexity computation + loss curve plots
│   └── inference.py       # load saved model + generate text
├── train.py               # entry point — runs full pipeline
├── notebook.ipynb         # Colab-ready notebook with inline plots
├── requirements.txt
└── README.md
```

The project is split into focused modules so each concern is isolated. `train.py` orchestrates everything — swap the dataset or model by changing one line in `config.yaml` without touching any source code.

---

## How It Works

### 1. Data pipeline

The dataset contains full-text Sci-Fi books as rows in a CSV. Empty rows and very short texts are dropped. Each book is tokenized to 512 tokens — sequences longer than 512 tokens are truncated, shorter ones are padded to maintain uniform batch shapes.

```python
# labels = input_ids for causal language modeling
# the model predicts each next token given all previous tokens
tokens["labels"] = tokens["input_ids"].copy()
```

### 2. Causal language modeling

GPT-2 is trained with a next-token prediction objective. Given the sequence `"The ship approached the"`, it learns to predict `"planet"`. Cross-entropy loss measures how surprised the model is by each actual next token. Lower loss = model has learned the domain's patterns.

### 3. Training setup

```
Effective batch size = per_device_batch (1) × gradient_accumulation (8) = 8
Learning rate        = 5e-5 with cosine decay + 100 warmup steps
Precision            = fp16 (halves memory, speeds up compute on T4)
Best checkpoint      = saved based on lowest eval loss
```

### 4. Evaluation

**Perplexity** = `exp(eval_loss)`. Measures how surprised the model is by held-out text. Lower perplexity = model has learned the domain's patterns better.

- Random model: perplexity ~1000+
- Pretrained GPT-2 Medium on general text: ~30–50
- After Sci-Fi fine-tuning: drops further toward the domain distribution

### 5. Inference

The saved model is loaded and wrapped in a HuggingFace `pipeline` for clean text generation. Temperature and top-p sampling control the creativity vs coherence tradeoff.

---

## Quickstart

### Option A — Colab (recommended)

1. Open `notebook.ipynb` in Google Colab
2. Set runtime to **T4 GPU** — Runtime → Change runtime type → T4
3. Run all cells top to bottom
4. Training completes in ~14 minutes

### Option B — Local

```bash
git clone https://github.com/aieng-abdullah/LLM-Fine-Tuning-Pipeline
cd gpt2-finetune
pip install -r requirements.txt
python train.py
```

### Inference only

```python
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model     = AutoModelForCausalLM.from_pretrained("outputs/finetuned-gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("outputs/finetuned-gpt2-medium")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

out = generator(
    "In a distant galaxy, a lone spacecraft",
    max_new_tokens=100,
    temperature=0.8,
    do_sample=True,
)
print(out[0]["generated_text"])
```

---

## Key Engineering Decisions

**Why not use LoRA/QLoRA here?**
GPT-2 Medium at 345M parameters fits entirely in T4 VRAM with fp16. Quantization and adapter layers add complexity with no benefit at this scale. QLoRA makes sense for 7B+ models where full fine-tuning is impossible without it.

**Why gradient accumulation?**
T4 has 16GB VRAM. A batch of 8 full 512-token sequences doesn't fit. Accumulating gradients over 8 steps of batch-size-1 achieves the same effective batch mathematically — without OOM errors.

**Why cosine learning rate?**
Cosine decay smoothly reduces the learning rate over training, preventing the model from overshooting optimal weights at the end. Combined with warmup steps, it stabilizes early training when gradients are noisy.

**Why save best checkpoint by eval loss?**
Training loss always decreases. Eval loss is the honest signal — it measures performance on data the model never saw. Saving the checkpoint with lowest eval loss guards against overfitting.

**Why full fine-tuning instead of LoRA on GPT-2?**
LoRA reduces trainable parameters by injecting low-rank matrices into attention layers. For a 345M model on a 16GB GPU, there is no memory pressure that requires this tradeoff. Full fine-tuning updates all weights and produces better domain adaptation when compute allows it.

---

## Configuration

All hyperparameters live in `configs/config.yaml`. To experiment with a different model:

```yaml
model:
  name: "gpt2-large"        # or gpt2-xl, distilgpt2

data:
  csv_path: "./your-dataset/data.csv"   # any CSV with a 'text' column
```

---

## Outputs

After training, `outputs/` contains:

```
outputs/
├── finetuned-gpt2-medium/    # full model weights
├── loss_curves.png           # train + eval loss over steps
├── eval_report.json          # eval loss + perplexity
└── metrics_history.json      # raw step-by-step loss logs
```

---

## References

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., 2019
- [Sci-Fi Books dataset](https://huggingface.co/datasets/stevez80/Sci-Fi-Books-gutenberg)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [HuggingFace Trainer docs](https://huggingface.co/docs/transformers/main_classes/trainer)