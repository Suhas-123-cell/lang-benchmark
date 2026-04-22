#!/usr/bin/env python3
"""
run_benchmark.py
================
Run the full Indic Language Benchmark locally on MPS/CUDA/CPU.
Loads one model at a time to stay within 16GB unified memory.

Usage:
    python3 run_benchmark.py                    # Run all models, all tasks
    python3 run_benchmark.py --model sarvam-2b  # Run single model
    python3 run_benchmark.py --samples 10       # Quick test with 10 samples
"""

import argparse
import os
import sys
import time

# Ensure src is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Load .env file for HF_TOKEN
_env_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()
    if os.environ.get("HF_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
        print(f"🔑 HF_TOKEN loaded from .env")

from src.data_loader import load_indicqa, load_indicmmlu, load_code_mixed_qa
from src.model_loader import (
    load_model_and_tokenizer, unload_model, get_device, MODEL_REGISTRY,
)
from src.inference import run_full_benchmark
from src.metrics import evaluate_results, format_metrics_report
from src.results import (
    load_all_results, build_leaderboard, build_summary_table,
    save_leaderboard_markdown, generate_all_charts,
)


def main():
    parser = argparse.ArgumentParser(description="Indic Language Benchmark Suite")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Run a specific model (sarvam-2b, gemma-2b, llama-3.2-1b). Default: all"
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Run a specific task (reading, math, codemixed). Default: all"
    )
    parser.add_argument(
        "--samples", type=int, default=50,
        help="Max samples per language for IndicQA/IndicMMLU (default: 50)"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--skip-inference", action="store_true",
        help="Skip inference, only generate leaderboard from existing results"
    )
    args = parser.parse_args()

    device = get_device()
    print(f"\n🚀 Indic Language Benchmark Suite")
    print(f"   Device: {device.upper()}")
    print(f"   Samples per language: {args.samples}")
    print(f"   Results dir: {args.results_dir}")
    print("=" * 60)

    raw_dir = os.path.join(args.results_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "scores"), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "figures"), exist_ok=True)

    if not args.skip_inference:
        # ─── Load Datasets ────────────────────────────────────────────
        datasets = {}

        task_filter = args.task
        if task_filter is None or task_filter == "reading":
            print("\n📖 Loading IndicQA...")
            datasets["reading_comprehension"] = load_indicqa(
                languages=["hindi", "bengali", "tamil"],
                max_samples_per_lang=args.samples,
            )

        if task_filter is None or task_filter == "math":
            print("\n🔢 Loading IndicMMLU-Pro...")
            datasets["math_reasoning"] = load_indicmmlu(
                languages=["hindi", "bengali", "tamil"],
                max_samples_per_lang=args.samples,
            )

        if task_filter is None or task_filter == "codemixed":
            print("\n🗣️  Loading Code-Mixed QA...")
            datasets["code_mixed_qa"] = load_code_mixed_qa()

        # ─── Run Inference Per Model ──────────────────────────────────
        models_to_run = [args.model] if args.model else list(MODEL_REGISTRY.keys())

        for model_key in models_to_run:
            if model_key not in MODEL_REGISTRY:
                print(f"❌ Unknown model: {model_key}")
                continue

            start = time.time()
            config_entry = MODEL_REGISTRY[model_key]

            try:
                model, tokenizer, config = load_model_and_tokenizer(
                    model_key, quantize=(device == "cuda")
                )
            except Exception as e:
                print(f"❌ Failed to load {model_key}: {e}")
                print("   Skipping this model...")
                continue

            # Run all tasks
            results = run_full_benchmark(
                model=model,
                tokenizer=tokenizer,
                model_key=model_key,
                model_name=config.name,
                datasets=datasets,
                results_dir=raw_dir,
                use_fewshot=True,
            )

            # Print metrics per task
            for task_name, task_results in results.items():
                metrics = evaluate_results(task_results, task_name)
                print(format_metrics_report(metrics, config.name))

            elapsed = time.time() - start
            print(f"\n⏱️  {config.name} completed in {elapsed:.0f}s")

            # Unload before next model
            unload_model(model, tokenizer)

    # ─── Generate Leaderboard ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 Generating Leaderboard")
    print("=" * 60)

    all_results = load_all_results(raw_dir)
    if not all_results:
        print("❌ No results found. Run inference first.")
        return

    leaderboard = build_leaderboard(all_results)

    # Summary
    summary = build_summary_table(leaderboard)
    print("\n🏆 LEADERBOARD (Primary Metric per Task):")
    print(summary.to_markdown())

    # Detailed
    print("\n📋 DETAILED RESULTS:")
    print(leaderboard.to_markdown(index=False))

    # Save
    save_leaderboard_markdown(leaderboard, os.path.join(args.results_dir, "leaderboard.md"))

    # Charts
    try:
        generate_all_charts(leaderboard, os.path.join(args.results_dir, "figures"))
    except Exception as e:
        print(f"⚠️  Chart generation failed: {e}")

    print("\n🎉 Benchmark complete!")
    print(f"   📂 Results: {args.results_dir}/")
    print(f"   📊 Leaderboard: {args.results_dir}/leaderboard.md")
    print(f"   📈 Charts: {args.results_dir}/figures/")


if __name__ == "__main__":
    main()
