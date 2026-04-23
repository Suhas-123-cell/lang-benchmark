"""
🏆 Indic Language Benchmark — HuggingFace Space
================================================
Interactive leaderboard comparing Sarvam-2B vs Gemma-2B across
reading comprehension, math reasoning, and code-mixed QA in
Hindi, Bengali, and Tamil.
"""

import json
import os
from pathlib import Path

import gradio as gr
import pandas as pd

# ─── Load Data ────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "data" / "results.json"
ASSETS_PATH = Path(__file__).parent / "assets"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    DATA = json.load(f)


# ─── Build DataFrames ────────────────────────────────────────────────────────

def build_overall_leaderboard():
    """Build the main summary leaderboard DataFrame."""
    rows = []
    for entry in DATA["overall_leaderboard"]:
        rows.append({
            "Rank": "",
            "Model": entry["model"],
            "📖 Reading Comp (F1%)": entry["reading_comprehension_f1"],
            "🔢 Math Reasoning (Acc%)": entry["math_reasoning_acc"],
            "🗣️ Code-Mixed QA (F1%)": entry["code_mixed_qa_f1"],
            "⭐ Average": entry["average"],
        })
    # Sort by average descending and assign ranks
    rows.sort(key=lambda x: x["⭐ Average"], reverse=True)
    for i, row in enumerate(rows):
        row["Rank"] = f"🥇" if i == 0 else f"🥈" if i == 1 else f"#{i+1}"
    return pd.DataFrame(rows)


def build_reading_comp_table():
    """Build detailed reading comprehension results."""
    rows = []
    for entry in DATA["detailed_results"]["reading_comprehension"]:
        rows.append({
            "Model": entry["model"],
            "Language": entry["language"].title(),
            "Exact Match (%)": entry["em"],
            "F1 Score (%)": entry["f1"],
        })
    return pd.DataFrame(rows)


def build_math_table():
    """Build detailed math reasoning results."""
    rows = []
    for entry in DATA["detailed_results"]["math_reasoning"]:
        rows.append({
            "Model": entry["model"],
            "Language": entry["language"].title(),
            "Accuracy (%)": entry["accuracy"],
        })
    return pd.DataFrame(rows)


def build_code_mixed_table():
    """Build detailed code-mixed QA results."""
    rows = []
    for entry in DATA["detailed_results"]["code_mixed_qa"]:
        rows.append({
            "Model": entry["model"],
            "Language": entry["language"].title(),
            "Exact Match (%)": entry["em"],
            "F1 Score (%)": entry["f1"],
            "BLEU (%)": entry["bleu"],
        })
    return pd.DataFrame(rows)


def build_model_info_table():
    """Build model information table."""
    rows = []
    for model in DATA["models"]:
        rows.append({
            "Model": model["name"],
            "Parameters": model["params"],
            "Indic-Focused": "✅" if model["indic_focused"] else "❌",
            "HuggingFace ID": model["hf_id"],
        })
    return pd.DataFrame(rows)


# ─── Markdown Content ────────────────────────────────────────────────────────

TITLE = """
<div style="text-align: center; padding: 20px 0;">
    <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">
        🏆 Indic Language Benchmark
    </h1>
    <p style="font-size: 1.1rem; color: #666; max-width: 700px; margin: 0 auto;">
        Evaluating multilingual LLMs on reading comprehension, math reasoning, and
        code-mixed QA across <strong>Hindi</strong>, <strong>Bengali</strong>, and <strong>Tamil</strong>
    </p>
    <p style="font-size: 0.9rem; color: #888; margin-top: 8px;">
        🖥️ All inference run locally on Apple Silicon (MPS) · Fully reproducible · Greedy decoding
    </p>
</div>
"""

KEY_FINDINGS = """
### 📊 Key Findings

| # | Finding | Detail |
|---|---------|--------|
| 1 | **Gemma-2B dominates reading comprehension** | 27.43% F1 vs Sarvam-2B's 1.87% — a **14.6×** gap. Gemma likely had more supervised QA data in pre-training. |
| 2 | **Sarvam-2B wins on Indic-specific tasks** | Math (14% vs 10%) and Code-Mixed QA (17.46% vs 8.56%) — **40–100%** better. Its custom tokenizer and Indic-focused pre-training data pay off. |
| 3 | **Code-mixing remains an open challenge** | Both models score F1 < 18% on Hinglish. Sarvam-2B's 2× advantage suggests better mixed-script handling, but huge room for improvement. |
| 4 | **Hindi > Tamil > Bengali** | Consistent across all tasks and models — directly correlates with web data availability. |
| 5 | **Base models have fundamental limits** | Both are non-instruction-tuned. Low EM scores (near 0% for Sarvam-2B on QA) reflect the gap between pre-training and task-specific capabilities. |
"""

METHODOLOGY = """
### 🔬 Evaluation Protocol

| Setting | Value |
|---------|-------|
| **Decoding** | Greedy (`do_sample=False`) — fully deterministic |
| **Prompting** | Few-shot (reading & math), Zero-shot (code-mixed) |
| **Text Normalization** | Indic-aware — Unicode normalization for Devanagari, Bengali, Tamil |
| **Context Truncation** | 1500 characters (reading comprehension) |
| **Max New Tokens** | 64 (reading), 16 (math), 48 (code-mixed) |
| **Device** | Apple Silicon MPS (float16) |

---

### 📚 Datasets

| Task | Dataset | Source | Languages | Samples/Model |
|------|---------|--------|-----------|:-------------:|
| 📖 Reading Comprehension | IndicQA | [AI4Bharat](https://huggingface.co/datasets/ai4bharat/IndicQA) | Hindi, Bengali, Tamil | 150 |
| 🔢 Math Reasoning | IndicMMLU-Pro | [LinguaLift](https://huggingface.co/datasets/LinguaLift/IndicMMLU-Pro) | Hindi, Bengali, Tamil | 150 |
| 🗣️ Code-Mixed QA | Hinglish QA | Hand-crafted (50 samples) | Hinglish | 50 |

---

### 📏 Metrics

| Metric | Tasks | What It Measures |
|--------|-------|------------------|
| **Exact Match (EM)** | Reading Comp, Code-Mixed | Exact string match after normalization |
| **F1 Score** | Reading Comp, Code-Mixed | Token-level overlap — partial credit |
| **Accuracy** | Math Reasoning | Correct option letter (A, B, C…) |
| **BLEU** | Code-Mixed | N-gram precision with brevity penalty |

---

### 🤖 Models

| Model | Params | Indic-Focused | HuggingFace |
|-------|:------:|:-------------:|-------------|
| **Sarvam-2B** | 2B | ✅ 10 Indic languages | [`sarvamai/sarvam-2b-v0.5`](https://huggingface.co/sarvamai/sarvam-2b-v0.5) |
| **Gemma-2B** | 2B | ❌ General-purpose | [`google/gemma-2b`](https://huggingface.co/google/gemma-2b) |
"""

CITATION = """
### 📝 Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{indic-language-benchmark-2026,
  title={Indic Language Benchmark: Evaluating Sarvam-2B vs Gemma-2B on Hindi, Bengali, and Tamil},
  author={Suhas Dev},
  year={2026},
  url={https://huggingface.co/spaces/YOUR_USERNAME/indic-llm-benchmark}
}
```

### 🙏 Acknowledgments

- [Sarvam AI](https://sarvam.ai) for Sarvam-2B
- [AI4Bharat](https://ai4bharat.org) for IndicQA
- [LinguaLift](https://huggingface.co/LinguaLift) for IndicMMLU-Pro
- [Google](https://ai.google) for Gemma-2B
"""


# ─── Custom CSS ───────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* Overall theme */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* Clean table styling */
table {
    border-collapse: collapse !important;
    width: 100% !important;
}

table th {
    background-color: #f8f9fa !important;
    color: #333 !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    text-align: center !important;
    border-bottom: 2px solid #dee2e6 !important;
}

table td {
    padding: 8px 14px !important;
    text-align: center !important;
    border-bottom: 1px solid #eee !important;
}

table tr:hover {
    background-color: #f8f9fa !important;
}

/* Tab styling */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 1rem !important;
}

.tab-nav button.selected {
    border-bottom: 3px solid #FF6B35 !important;
    color: #FF6B35 !important;
}
"""


# ─── Build the App ────────────────────────────────────────────────────────────

def create_app():
    with gr.Blocks(
        title="🏆 Indic Language Benchmark",
    ) as demo:

        # ── Header ──
        gr.HTML(TITLE)

        # ── Tabs ──
        with gr.Tabs():

            # ═══════════════════════════════════════════════════════════
            # TAB 1: LEADERBOARD
            # ═══════════════════════════════════════════════════════════
            with gr.TabItem("🏆 Leaderboard", id="leaderboard"):

                gr.Markdown("### Overall Leaderboard")
                gr.Markdown("*Primary metric per task. Sorted by average score.*")

                overall_df = build_overall_leaderboard()
                gr.Dataframe(
                    value=overall_df,
                    headers=list(overall_df.columns),
                    interactive=False,
                    wrap=True,
                    row_count=(len(overall_df), "fixed"),
                )

                gr.Markdown("---")
                gr.Markdown(KEY_FINDINGS)

                # ── Detailed breakdowns ──
                gr.Markdown("---")
                gr.Markdown("### Detailed Breakdown by Task")

                with gr.Accordion("📖 Reading Comprehension (IndicQA — F1%)", open=False):
                    rc_df = build_reading_comp_table()
                    gr.Dataframe(
                        value=rc_df,
                        headers=list(rc_df.columns),
                        interactive=False,
                        wrap=True,
                    )

                with gr.Accordion("🔢 Math Reasoning (IndicMMLU-Pro — Accuracy%)", open=False):
                    math_df = build_math_table()
                    gr.Dataframe(
                        value=math_df,
                        headers=list(math_df.columns),
                        interactive=False,
                        wrap=True,
                    )

                with gr.Accordion("🗣️ Code-Mixed QA (Hinglish)", open=False):
                    cm_df = build_code_mixed_table()
                    gr.Dataframe(
                        value=cm_df,
                        headers=list(cm_df.columns),
                        interactive=False,
                        wrap=True,
                    )

            # ═══════════════════════════════════════════════════════════
            # TAB 2: VISUALIZATIONS
            # ═══════════════════════════════════════════════════════════
            with gr.TabItem("📊 Visualizations", id="visualizations"):

                gr.Markdown("### 📊 Benchmark Visualizations")
                gr.Markdown("*Auto-generated charts from benchmark results.*")

                with gr.Row():
                    model_comp_path = str(ASSETS_PATH / "model_comparison.png")
                    if os.path.exists(model_comp_path):
                        gr.Image(
                            value=model_comp_path,
                            label="Model Comparison (Bar Chart)",
                            height=400,
                        )

                with gr.Row():
                    with gr.Column():
                        heatmap_path = str(ASSETS_PATH / "language_heatmap.png")
                        if os.path.exists(heatmap_path):
                            gr.Image(
                                value=heatmap_path,
                                label="Language Performance Heatmap",
                                height=350,
                            )
                    with gr.Column():
                        radar_path = str(ASSETS_PATH / "radar_chart.png")
                        if os.path.exists(radar_path):
                            gr.Image(
                                value=radar_path,
                                label="Model Capability Radar",
                                height=350,
                            )

            # ═══════════════════════════════════════════════════════════
            # TAB 3: ABOUT / METHODOLOGY
            # ═══════════════════════════════════════════════════════════
            with gr.TabItem("📖 About", id="about"):
                gr.Markdown(METHODOLOGY)
                gr.Markdown("---")
                gr.Markdown(CITATION)

    return demo


# ─── Launch ───────────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.blue,
    font=gr.themes.GoogleFont("Inter"),
)

if __name__ == "__main__":
    demo = create_app()
    demo.launch(theme=THEME, css=CUSTOM_CSS)
else:
    # HF Spaces imports app.py as a module
    demo = create_app()
    demo.launch(theme=THEME, css=CUSTOM_CSS)
