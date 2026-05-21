# Mitra — Tenglish Friend Chatbot 🎬🎵

Fine-tune Sarvam-2B using MLX LoRA on Apple Silicon to create a warm
Tenglish (Telugu-English) friend chatbot that recommends movies and
songs based on mood.

## Project Structure

```
mitra/
├── setup.sh              # Step 1: Environment setup
├── generate_dataset.py   # Step 2: Generate 600 conversations
├── chat_mitra.py         # Step 4: Interactive chat
├── eval_mitra.py         # Step 5: Base vs fine-tuned comparison
├── push_to_hub.py        # Step 6: Upload to HuggingFace
├── MODEL_CARD.md         # HuggingFace model card
├── data/
│   ├── train.jsonl       # 540 training conversations
│   └── valid.jsonl       # 60 validation conversations
├── adapters/             # LoRA weights (created during training)
└── results/
    ├── eval_results.txt  # Comparison results
    └── eval_results.json
```

---

## Step 1 — Environment Setup

```bash
# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install mlx-lm huggingface_hub datasets

# Login to HuggingFace
huggingface-cli login

# Test model loads
python3 -c "
from mlx_lm import load, generate
model, tokenizer = load('sarvamai/sarvam-2b-v0.5')
r = generate(model, tokenizer, prompt='Hyderabad is a beautiful city', max_tokens=50)
print(r)
"
```

## Step 2 — Generate Dataset

```bash
python3 generate_dataset.py
# Creates data/train.jsonl (540) and data/valid.jsonl (60)
```

## Step 3 — Fine-tune with MLX LoRA

```bash
python3 -m mlx_lm.lora \
  --model sarvamai/sarvam-2b-v0.5 \
  --train --data ./data --adapter-path ./adapters \
  --batch-size 2 --num-layers 16 \
  --iters 1000 --learning-rate 1e-5 \
  --steps-per-eval 100 --save-every 100 --val-batches 5 \
  --max-seq-length 2048
```

**Expected**: ~2-3 hours on M5 16GB. Watch the training loss decrease.

## Step 4 — Chat with Mitra

```bash
# Fine-tuned (Mitra personality)
python3 chat_mitra.py

# Base model (for comparison)
python3 chat_mitra.py --base
```

## Step 5 — Evaluate

```bash
python3 eval_mitra.py
# Results saved to results/eval_results.txt
```

## Step 6 — Push to HuggingFace

Edit `push_to_hub.py` → set `HF_USERNAME` and `MODEL_CARD.md` → set `[YOUR_*]` fields.

```bash
python3 push_to_hub.py
```

---

## Step 7 — Sarvam Email

**Subject:** Native Telugu Speaker — Fine-tuned Sarvam-2B for Tenglish Conversations (MLX, Apple Silicon)

> Hi Sarvam team,
>
> I'm [YOUR_NAME], a student at IIIT and a native Telugu speaker. I fine-tuned
> Sarvam-2B using LoRA on MLX (Apple Silicon M5) to build "Mitra" — a Tenglish
> friend chatbot that recommends movies and songs based on mood.
>
> I curated 600 Tenglish conversations, validated each one using my native
> intuition, and trained the model entirely locally. The fine-tuned model shows
> clear improvement in generating natural, contextual Tenglish vs the base model.
>
> HuggingFace: https://huggingface.co/[YOUR_HF_USERNAME]/mitra-tenglish-sarvam2b
> Eval results: [link to eval_results.txt]
>
> I'd love to contribute to Sarvam's Indic language work as an intern. Happy to
> discuss my approach and findings.
>
> Best,
> [YOUR_NAME]
