#!/usr/bin/env python3
"""Evaluate base Sarvam-2B vs Mitra (fine-tuned) on 5 test prompts."""
import os, json
from datetime import datetime
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

TEST_PROMPTS = [
    "yaar chala bore avutunna",
    "bro exam fail ayanu, chala bad ga undi",
    "oka sad movie suggest cheyyi",
    "yaar happy songs emi vinali",
    "bro chill cheyyadaniki emi chusukovali",
]

SAMPLER = make_sampler(temp=0.7, top_p=0.9)

def run_eval(model, tokenizer, label):
    results = []
    for prompt in TEST_PROMPTS:
        formatted = f"<s>[INST] {prompt} [/INST]"
        response = generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=200,
            sampler=SAMPLER,
        )
        response = response.strip()
        if "</s>" in response:
            response = response.split("</s>")[0].strip()
        results.append({"prompt": prompt, "response": response})
        print(f"  [{label}] {prompt}")
        print(f"  → {response}\n")
    return results


def main():
    model_path = "sarvamai/sarvam-2b-v0.5"
    adapter_path = "./adapters"

    # --- BASE MODEL ---
    print("=" * 60)
    print("  Loading BASE model (no adapter)")
    print("=" * 60)
    model, tokenizer = load(model_path)
    base_results = run_eval(model, tokenizer, "BASE")
    del model, tokenizer

    # --- MITRA (fine-tuned) ---
    print("=" * 60)
    print("  Loading MITRA model (with LoRA adapter)")
    print("=" * 60)
    if not os.path.exists(adapter_path):
        print(f"ERROR: Adapter not found at {adapter_path}")
        print("Run fine-tuning first! See README.md Step 3.")
        mitra_results = [{"prompt": p, "response": "[NOT TRAINED YET]"} for p in TEST_PROMPTS]
    else:
        model, tokenizer = load(model_path, adapter_path=adapter_path)
        mitra_results = run_eval(model, tokenizer, "MITRA")

    # --- Save results ---
    output = []
    output.append("=" * 70)
    output.append("  MITRA EVALUATION — Base vs Fine-tuned Sarvam-2B")
    output.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    output.append("=" * 70)
    output.append("")

    for i, prompt in enumerate(TEST_PROMPTS):
        output.append(f"PROMPT {i+1}: {prompt}")
        output.append("-" * 50)
        output.append(f"BASE:  {base_results[i]['response']}")
        output.append(f"MITRA: {mitra_results[i]['response']}")
        output.append("")

    result_text = "\n".join(output)
    os.makedirs("results", exist_ok=True)

    with open("results/eval_results.txt", "w", encoding="utf-8") as f:
        f.write(result_text)

    with open("results/eval_results.json", "w", encoding="utf-8") as f:
        json.dump({"base": base_results, "mitra": mitra_results}, f, indent=2, ensure_ascii=False)

    print("\n" + result_text)
    print(f"\nResults saved to results/eval_results.txt")


if __name__ == "__main__":
    main()
