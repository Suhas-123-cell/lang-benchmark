---
language:
  - te
  - en
license: mit
library_name: mlx
base_model: sarvamai/sarvam-2b-v0.5
tags:
  - tenglish
  - telugu-english
  - code-mixed
  - indic
  - conversational
  - lora
  - mlx
  - apple-silicon
  - sarvam
model_type: llama
pipeline_tag: text-generation
---

# Mitra — Tenglish Friend Chatbot 🎬🎵

**Mitra** is a LoRA fine-tuned version of [Sarvam-2B](https://huggingface.co/sarvamai/sarvam-2b-v0.5) that talks like a warm Telugu college friend — recommending movies and songs based on your mood in natural **Tenglish** (Telugu-English code-mixed).

> Built by Maddali Suhas Choudary, IIIT Senapati, Manipur

## What is Mitra?

Mitra is your Tenglish buddy who:
- 🎬 Suggests Telugu/Hindi movies with personal reasons, not boring lists
- 🎵 Recommends songs that match your mood
- 💬 Talks naturally in Romanized Tenglish like a real college friend
- 🤗 Provides empathetic support when you're stressed or sad

## Dataset

- **600 multi-turn Tenglish conversations** across 6 mood categories:
  - Bored | Stressed/Exam Pressure | Happy/Excited | Sad/Low Mood | Movie Suggestions | Song Suggestions
- **Format**: Llama-2 instruction template (`<s>[INST] ... [/INST] ... </s>`)
- **Language**: Romanized Telugu-English code-mixed (Tenglish)
- **Source**: generated from templates by `generate_dataset.py` (seed 42), 540 train / 60 validation
- All movie/song recommendations are real Telugu/Hindi titles

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | `sarvamai/sarvam-2b-v0.5` |
| Method | LoRA (Low-Rank Adaptation) |
| Framework | MLX (Apple Silicon native) |
| Hardware | MacBook M5, 16GB unified memory |
| LoRA Rank | 8 |
| LoRA Layers | 16 |
| Learning Rate | 1e-5 |
| Batch Size | 2 |
| Iterations | 1000 |

## Example: Base vs Fine-tuned

### Prompt: "yaar chala bore avutunna"

Outputs copied from [results/eval_results.txt](results/eval_results.txt) (5 prompts, not a scored benchmark).

**Base Sarvam-2B:**
> येर चल बुतूअन

**Mitra (fine-tuned):**
> arey bore na? oka walk vellu, fresh air help chestundi

### Prompt: "bro exam fail ayanu, chala bad ga undi"

**Base Sarvam-2B:** *(unrelated text about national film awards, repeated)*

**Mitra (fine-tuned):**
> arey bro, first deep breath teesko. fail aithe aiindi, next time kill cheddham

## How to Use

### With MLX (Apple Silicon)

The adapter weights are not committed or published; train them with the steps in
[README.md](README.md), which write them to `./adapters`.

```python
from mlx_lm import load, generate

model, tokenizer = load(
    "sarvamai/sarvam-2b-v0.5",
    adapter_path="./adapters"
)

prompt = "<s>[INST] bro chill movie suggest chey [/INST]"
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
```

## Limitations

- Trained on 600 conversations — limited domain coverage (moods + movie/song recommendations)
- Romanized Tenglish only (no Telugu script)
- Movie/song knowledge limited to the training set
- Base model (Sarvam-2B v0.5) is a pre-training checkpoint, not a fully trained model

## Citation

```bibtex
@misc{mitra-tenglish-2025,
  title={Mitra: A Tenglish Friend Chatbot Fine-tuned on Sarvam-2B},
  author={Maddali Suhas Choudary},
  year={2025},
  url={https://github.com/Suhas-123-cell/lang-benchmark}
}
```
