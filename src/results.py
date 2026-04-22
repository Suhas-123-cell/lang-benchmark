"""
Results Aggregation & Visualization Module
==========================================
Aggregates benchmark results from all models and tasks into:
1. A structured leaderboard DataFrame
2. Markdown tables for README and HuggingFace
3. Comparison charts (bar charts, heatmaps, radar charts)
"""

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab/server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import evaluate_results


# ─── Color Palette ────────────────────────────────────────────────────────────

# Curated palette inspired by Sarvam AI branding
COLORS = {
    "sarvam-2b": "#FF6B35",    # Sarvam orange
    "gemma-2b": "#4285F4",     # Google blue
    "llama-3.2-1b": "#7B2FF7", # Meta purple
}

TASK_COLORS = {
    "reading_comprehension": "#2ECC71",
    "math_reasoning": "#E74C3C",
    "code_mixed_qa": "#F39C12",
}

LANG_COLORS = {
    "hindi": "#FF9933",     # Saffron
    "bengali": "#DC3545",   # Crimson
    "tamil": "#138808",     # Green
    "hinglish": "#6C757D",  # Gray
}


# ─── Results Loading ─────────────────────────────────────────────────────────

def load_all_results(results_dir: str = "results/raw") -> List[Dict]:
    """Load all JSONL result files from the raw results directory."""
    all_results = []

    if not os.path.exists(results_dir):
        print(f"⚠️  Results directory '{results_dir}' not found.")
        return []

    for filename in sorted(os.listdir(results_dir)):
        if not filename.endswith(".jsonl"):
            continue

        filepath = os.path.join(results_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

    print(f"📂 Loaded {len(all_results)} results from {results_dir}")
    return all_results


# ─── Leaderboard Construction ────────────────────────────────────────────────

def build_leaderboard(
    all_results: List[Dict],
) -> pd.DataFrame:
    """
    Build a leaderboard DataFrame from raw results.

    The leaderboard shows the primary metric for each model × task × language
    combination.

    Returns:
        DataFrame with columns: Model, Task, Language, Metric, Score
    """
    rows = []

    # Group results by model and task
    grouped = {}
    for r in all_results:
        key = (r["model"], r["task"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    for (model, task), results in grouped.items():
        metrics = evaluate_results(results, task)

        # Extract primary metric based on task
        if task == "reading_comprehension":
            # EM and F1
            em_data = metrics.get("exact_match", {})
            f1_data = metrics.get("f1", {})

            # Overall
            rows.append({
                "Model": model, "Task": task, "Language": "Overall",
                "EM (%)": em_data.get("exact_match", 0),
                "F1 (%)": f1_data.get("f1", 0),
                "Accuracy (%)": "-", "BLEU (%)": "-",
            })

            # Per language
            for lang in em_data.get("per_language", {}):
                rows.append({
                    "Model": model, "Task": task, "Language": lang,
                    "EM (%)": em_data["per_language"].get(lang, 0),
                    "F1 (%)": f1_data.get("per_language", {}).get(lang, 0),
                    "Accuracy (%)": "-", "BLEU (%)": "-",
                })

        elif task == "math_reasoning":
            acc_data = metrics.get("accuracy", {})

            rows.append({
                "Model": model, "Task": task, "Language": "Overall",
                "EM (%)": "-", "F1 (%)": "-",
                "Accuracy (%)": acc_data.get("accuracy", 0),
                "BLEU (%)": "-",
            })

            for lang in acc_data.get("per_language", {}):
                rows.append({
                    "Model": model, "Task": task, "Language": lang,
                    "EM (%)": "-", "F1 (%)": "-",
                    "Accuracy (%)": acc_data["per_language"].get(lang, 0),
                    "BLEU (%)": "-",
                })

        elif task == "code_mixed_qa":
            em_data = metrics.get("exact_match", {})
            f1_data = metrics.get("f1", {})
            bleu_data = metrics.get("bleu", {})

            rows.append({
                "Model": model, "Task": task, "Language": "Overall",
                "EM (%)": em_data.get("exact_match", 0),
                "F1 (%)": f1_data.get("f1", 0),
                "Accuracy (%)": "-",
                "BLEU (%)": bleu_data.get("bleu", 0),
            })

    df = pd.DataFrame(rows)
    return df


def build_summary_table(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """
    Build a high-level summary table showing overall scores per model per task.

    This is the main "leaderboard" table for the README.
    """
    overall = leaderboard[leaderboard["Language"] == "Overall"].copy()

    # Determine primary metric per task
    def get_primary_score(row):
        if row["Task"] == "reading_comprehension":
            return row["F1 (%)"]
        elif row["Task"] == "math_reasoning":
            return row["Accuracy (%)"]
        elif row["Task"] == "code_mixed_qa":
            return row["F1 (%)"]
        return 0

    overall["Primary Score (%)"] = overall.apply(get_primary_score, axis=1)

    # Pivot to model × task
    pivot = overall.pivot_table(
        index="Model",
        columns="Task",
        values="Primary Score (%)",
        aggfunc="first",
    )

    # Calculate average across tasks
    numeric_cols = pivot.select_dtypes(include=[np.number]).columns
    pivot["Average"] = pivot[numeric_cols].mean(axis=1).round(2)

    # Sort by average descending
    pivot = pivot.sort_values("Average", ascending=False)

    return pivot


# ─── Markdown Generation ─────────────────────────────────────────────────────

def leaderboard_to_markdown(
    leaderboard: pd.DataFrame,
    title: str = "🏆 Indic Language Benchmark Leaderboard",
) -> str:
    """Convert leaderboard DataFrame to a formatted Markdown table."""
    lines = [f"## {title}", ""]

    # Summary table
    summary = build_summary_table(leaderboard)
    lines.append("### Overall Results (Primary Metric per Task)")
    lines.append("")
    lines.append(summary.to_markdown())
    lines.append("")

    # Detailed results per task
    for task in leaderboard["Task"].unique():
        task_data = leaderboard[leaderboard["Task"] == task]
        task_name = task.replace("_", " ").title()
        lines.append(f"### {task_name}")
        lines.append("")
        lines.append(task_data.to_markdown(index=False))
        lines.append("")

    return "\n".join(lines)


def save_leaderboard_markdown(
    leaderboard: pd.DataFrame,
    output_path: str = "results/leaderboard.md",
):
    """Save the leaderboard as a Markdown file."""
    md_content = leaderboard_to_markdown(leaderboard)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"📝 Leaderboard saved to {output_path}")


# ─── Visualization ────────────────────────────────────────────────────────────

def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial", "sans-serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def plot_grouped_bar_chart(
    leaderboard: pd.DataFrame,
    output_path: str = "results/figures/model_comparison.png",
):
    """
    Create a grouped bar chart: Model × Task comparison.

    Shows the primary metric for each model across all tasks.
    """
    setup_plot_style()

    summary = build_summary_table(leaderboard)
    task_cols = [c for c in summary.columns if c != "Average"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(task_cols))
    width = 0.25
    multiplier = 0

    for model_name in summary.index:
        model_key = model_name.lower().replace(" ", "-").replace(".", "")
        color = COLORS.get(model_key, f"C{multiplier}")

        scores = []
        for task in task_cols:
            val = summary.loc[model_name, task]
            scores.append(float(val) if isinstance(val, (int, float, np.number)) else 0)

        offset = width * multiplier
        bars = ax.bar(x + offset, scores, width, label=model_name, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        multiplier += 1

    ax.set_xlabel("Task")
    ax.set_ylabel("Score (%)")
    ax.set_title("🏆 Indic Language Benchmark — Model Comparison", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([t.replace("_", " ").title() for t in task_cols], rotation=15)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Bar chart saved to {output_path}")


def plot_language_heatmap(
    leaderboard: pd.DataFrame,
    output_path: str = "results/figures/language_heatmap.png",
):
    """
    Create a heatmap: Model × Language performance.

    Shows how each model performs across different Indic languages.
    """
    setup_plot_style()

    # Filter for per-language results (not "Overall")
    lang_data = leaderboard[leaderboard["Language"] != "Overall"].copy()

    if lang_data.empty:
        print("⚠️  No per-language data available for heatmap.")
        return

    # Extract primary score
    def get_score(row):
        for col in ["F1 (%)", "Accuracy (%)", "EM (%)"]:
            val = row.get(col, "-")
            if val != "-" and isinstance(val, (int, float)):
                return float(val)
        return 0.0

    lang_data["Score"] = lang_data.apply(get_score, axis=1)

    # Pivot: Model × Language
    pivot = lang_data.pivot_table(
        index="Model",
        columns="Language",
        values="Score",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=50,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Score (%)"},
    )
    ax.set_title("🌍 Performance Across Languages", fontweight="bold")
    ax.set_ylabel("Model")
    ax.set_xlabel("Language")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Heatmap saved to {output_path}")


def plot_radar_chart(
    leaderboard: pd.DataFrame,
    output_path: str = "results/figures/radar_chart.png",
):
    """
    Create a radar chart showing model capabilities across tasks.
    """
    setup_plot_style()

    summary = build_summary_table(leaderboard)
    task_cols = [c for c in summary.columns if c != "Average"]

    if len(task_cols) < 3:
        print("⚠️  Need at least 3 tasks for radar chart.")
        return

    categories = [t.replace("_", "\n").title() for t in task_cols]
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model_name in summary.index:
        model_key = model_name.lower().replace(" ", "-").replace(".", "")
        color = COLORS.get(model_key, "#666666")

        values = []
        for task in task_cols:
            val = summary.loc[model_name, task]
            values.append(float(val) if isinstance(val, (int, float, np.number)) else 0)
        values += values[:1]  # Close the polygon

        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("🎯 Model Capability Radar", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Radar chart saved to {output_path}")


def generate_all_charts(
    leaderboard: pd.DataFrame,
    output_dir: str = "results/figures",
):
    """Generate all visualization charts."""
    print("\n🎨 Generating visualizations...")
    plot_grouped_bar_chart(leaderboard, os.path.join(output_dir, "model_comparison.png"))
    plot_language_heatmap(leaderboard, os.path.join(output_dir, "language_heatmap.png"))
    plot_radar_chart(leaderboard, os.path.join(output_dir, "radar_chart.png"))
    print("✅ All charts generated!")


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def generate_full_report(
    results_dir: str = "results/raw",
    output_dir: str = "results",
):
    """
    Generate the complete benchmark report:
    1. Load all raw results
    2. Build leaderboard
    3. Save Markdown table
    4. Generate charts
    """
    print("\n" + "=" * 60)
    print("📋 Generating Full Benchmark Report")
    print("=" * 60)

    # Load results
    all_results = load_all_results(results_dir)
    if not all_results:
        print("❌ No results found. Run inference first.")
        return None

    # Build leaderboard
    leaderboard = build_leaderboard(all_results)
    print(f"\n📊 Leaderboard ({len(leaderboard)} rows):")
    print(leaderboard.to_string(index=False))

    # Save markdown
    save_leaderboard_markdown(leaderboard, os.path.join(output_dir, "leaderboard.md"))

    # Generate charts
    generate_all_charts(leaderboard, os.path.join(output_dir, "figures"))

    # Save scores as JSON
    scores_path = os.path.join(output_dir, "scores", "all_scores.json")
    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
    leaderboard.to_json(scores_path, orient="records", indent=2)
    print(f"💾 Scores saved to {scores_path}")

    return leaderboard


if __name__ == "__main__":
    generate_full_report()
