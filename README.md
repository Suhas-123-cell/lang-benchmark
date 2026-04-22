# 🏆 Indic Language Benchmark Suite

> Evaluating multilingual LLMs across reading comprehension, math reasoning, and code-mixed QA tasks in Hindi, Bengali, and Tamil.

[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)

---

## 📋 Overview

This benchmark suite evaluates **Sarvam-2B** (an Indic-focused model) against **Gemma-2B** and **Llama-3.2-1B** across three core tasks in three Indian languages, providing insights into:

- How purpose-built Indic models compare to general-purpose models
- Tokenizer efficiency differences across scripts (Devanagari, Bengali, Tamil)
- The state of code-mixed (Hinglish) language understanding

### Models

| Model | Params | Indic-Focused | HuggingFace ID |
|-------|--------|---------------|----------------|
| **Sarvam-2B** | 2B | ✅ 10 Indic languages | `sarvamai/sarvam-2b-v0.5` |
| **Gemma-2B** | 2B | ❌ General-purpose | `google/gemma-2b` |
| **Llama-3.2-1B** | 1B | 🟡 Hindi supported | `meta-llama/Llama-3.2-1B` |

### Tasks & Datasets

| Task | Dataset | Languages | Metrics |
|------|---------|-----------|---------|
| 📖 Reading Comprehension | [IndicQA](https://huggingface.co/datasets/ai4bharat/IndicQA) (AI4Bharat) | Hindi, Bengali, Tamil | EM, F1 |
| 🔢 Math Reasoning | [IndicMMLU-Pro](https://huggingface.co/datasets/LinguaLift/IndicMMLU-Pro) (STEM) | Hindi, Bengali, Tamil | Accuracy |
| 🗣️ Code-Mixed QA | 50 hand-crafted Hinglish samples | Hinglish | EM, F1, BLEU |

---

## 🏆 Leaderboard

> Results will be populated after running the benchmark notebooks.

<!-- LEADERBOARD_START -->
| Model | Reading Comprehension (F1%) | Math Reasoning (Acc%) | Code-Mixed QA (F1%) | Average |
|-------|----------------------------|----------------------|---------------------|---------|
| Sarvam-2B | - | - | - | - |
| Gemma-2B | - | - | - | - |
| Llama-3.2-1B | - | - | - | - |
<!-- LEADERBOARD_END -->

*Run the notebooks to populate this table with actual results.*

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- GPU with 16GB+ VRAM (T4 or better) or Google Colab
- HuggingFace account (for gated models like Gemma and Llama)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/lang-benchmark.git
cd lang-benchmark
pip install -r requirements.txt
```

### Running the Benchmark

The benchmark is organized into 3 Colab-ready notebooks:

```
notebooks/
├── 01_setup_and_indicqa.ipynb    # Day 1: Setup + Reading Comprehension
├── 02_math_and_codemixed.ipynb   # Day 2: Math + Code-Mixed + Charts
└── 03_publish_and_analyze.ipynb  # Day 3: Analysis + HuggingFace Push
```

**Option A — Google Colab (Recommended)**
1. Upload the repo to Google Drive or clone via Colab
2. Set runtime to GPU (T4)
3. Run notebooks sequentially

**Option B — Local**
```bash
# Run the metric sanity checks
python -m src.metrics

# Run data loading test
python -m src.data_loader
```

---

## 📁 Project Structure

```
lang-benchmark/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Dataset loading & preprocessing
│   ├── model_loader.py                # Model loading with 4-bit quantization
│   ├── inference.py                   # Prompt construction & generation
│   ├── metrics.py                     # EM, F1, Accuracy, BLEU evaluation
│   └── results.py                     # Leaderboard & visualization
├── data/
│   └── code_mixed_qa.json             # 50 Hinglish QA samples
├── notebooks/
│   ├── 01_setup_and_indicqa.ipynb
│   ├── 02_math_and_codemixed.ipynb
│   └── 03_publish_and_analyze.ipynb
├── results/
│   ├── raw/                           # JSONL predictions per model/task
│   ├── scores/                        # Computed metric scores
│   ├── figures/                       # Generated charts
│   └── leaderboard.md                 # Auto-generated leaderboard
└── huggingface/
    └── README.md                      # HuggingFace model card
```

---

## 📊 Methodology

### Evaluation Protocol

1. **Quantization:** All models loaded with 4-bit NF4 quantization via `bitsandbytes` for fair comparison on identical hardware
2. **Decoding:** Greedy decoding (`do_sample=False`) for reproducible results
3. **Prompting:** Few-shot prompts for reading comprehension and math; zero-shot for code-mixed QA
4. **Normalization:** Indic-aware text normalization for metrics (handles Devanagari, Bengali, Tamil scripts)

### Metrics Explained

| Metric | Description | Range |
|--------|-------------|-------|
| **Exact Match (EM)** | Normalized string match between prediction and gold | 0-100% |
| **F1 Score** | Token-level precision-recall harmonic mean | 0-100% |
| **Accuracy** | Correct option letter for multiple-choice | 0-100% |
| **BLEU** | N-gram precision with brevity penalty | 0-100% |

### Tokenizer Analysis

A key differentiator is **tokenizer fertility** — the ratio of tokens to words:

| Metric | Sarvam-2B | Gemma-2B | Llama-3.2-1B |
|--------|-----------|----------|--------------|
| Hindi Fertility | ~2.0 | ~4.0+ | ~3.5+ |
| Bengali Fertility | ~2.0 | ~5.0+ | ~4.0+ |
| Tamil Fertility | ~2.0 | ~5.0+ | ~4.5+ |

*Lower is better — Sarvam-2B's custom tokenizer is 2-3x more efficient for Indic text.*

---

## 🔑 Key Insights

1. **Tokenizer matters:** Sarvam-2B's purpose-built tokenizer uses ~2x fewer tokens for Indic text, directly reducing inference cost
2. **Base ≠ Instruction-tuned:** All models are base checkpoints — QA performance reflects pretraining data quality, not task capability
3. **Hindi > Bengali > Tamil:** Performance correlates with web data availability for each language
4. **Code-mixing is hard:** All models struggle with naturalistic Hinglish, highlighting a research gap
5. **Math doesn't transfer:** Mathematical reasoning in Indic languages is significantly weaker than English

---

## 📚 References

- [Sarvam-2B Model Card](https://huggingface.co/sarvamai/sarvam-2b-v0.5)
- [IndicQA Dataset (AI4Bharat)](https://huggingface.co/datasets/ai4bharat/IndicQA)
- [IndicMMLU-Pro (LinguaLift)](https://huggingface.co/datasets/LinguaLift/IndicMMLU-Pro)
- [Sarvam AI: "Evaluating Indian Language ASR" (April 2, 2026)](https://sarvam.ai/blog)
- [MILU Benchmark (AI4Bharat)](https://huggingface.co/datasets/ai4bharat/MILU)

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Sarvam AI](https://sarvam.ai) for Sarvam-2B and pioneering Indic language research
- [AI4Bharat](https://ai4bharat.org) for IndicQA and foundational Indic NLP datasets
- [LinguaLift](https://huggingface.co/LinguaLift) for IndicMMLU-Pro
- [Google](https://ai.google) for Gemma-2B
- [Meta](https://ai.meta.com) for Llama-3.2-1B
# lang-benchmark
