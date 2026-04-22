"""
Inference Module
================
Handles prompt construction and model inference for all three benchmark tasks.

Prompt templates are designed for base (non-instruction-tuned) models,
using simple few-shot formatting to guide output format.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm


# ─── Prompt Templates ────────────────────────────────────────────────────────

# Few-shot examples to guide base models on output format
READING_COMPREHENSION_TEMPLATE = """Read the context and answer the question concisely.

Context: {context}

Question: {question}

Answer:"""

MATH_REASONING_TEMPLATE = """Answer the following multiple-choice question by selecting the correct option letter.

Question: {question}

Options:
{options}

Answer:"""

CODE_MIXED_QA_TEMPLATE = """Answer the following question concisely.

Question: {question}

Answer:"""

# Few-shot prefix examples (helps base models understand the format)
READING_COMPREHENSION_FEWSHOT = """Read the context and answer the question concisely.

Context: भारत एक दक्षिण एशियाई देश है जो क्षेत्रफल के हिसाब से सातवां सबसे बड़ा देश है। भारत की राजधानी नई दिल्ली है।

Question: भारत की राजधानी क्या है?

Answer: नई दिल्ली

---

"""

MATH_REASONING_FEWSHOT = """Answer the following multiple-choice question by selecting the correct option letter.

Question: 2 + 2 = ?

Options:
A. 3
B. 4
C. 5
D. 6

Answer: B

---

"""


def build_prompt(
    sample: Dict,
    task: str,
    use_fewshot: bool = True,
) -> str:
    """
    Build an inference prompt for a given sample and task.

    Args:
        sample: Normalized sample dict from data_loader.
        task: Task type ("reading_comprehension", "math_reasoning", "code_mixed_qa").
        use_fewshot: Whether to prepend few-shot examples.

    Returns:
        Formatted prompt string.
    """
    if task == "reading_comprehension":
        prompt = READING_COMPREHENSION_TEMPLATE.format(
            context=sample["context"][:1500],  # Truncate long contexts
            question=sample["question"],
        )
        if use_fewshot:
            prompt = READING_COMPREHENSION_FEWSHOT + prompt

    elif task == "math_reasoning":
        options_str = "\n".join(sample.get("options", []))
        prompt = MATH_REASONING_TEMPLATE.format(
            question=sample["question"],
            options=options_str,
        )
        if use_fewshot:
            prompt = MATH_REASONING_FEWSHOT + prompt

    elif task == "code_mixed_qa":
        prompt = CODE_MIXED_QA_TEMPLATE.format(
            question=sample["question"],
        )
        # No few-shot for code-mixed — we want to test raw ability

    else:
        raise ValueError(f"Unknown task: {task}")

    return prompt


# ─── Inference Engine ─────────────────────────────────────────────────────────

@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.1,
    do_sample: bool = False,
    repetition_penalty: float = 1.2,
) -> str:
    """
    Generate a response from a model given a prompt.

    Uses greedy decoding by default for reproducibility.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        prompt: Input prompt string.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (only used if do_sample=True).
        do_sample: Whether to use sampling vs greedy decoding.
        repetition_penalty: Penalty for repeated tokens.

    Returns:
        Generated text (response only, prompt stripped).
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=False,
    )

    # Move to model's device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["do_sample"] = True
        generate_kwargs["top_p"] = 0.9
    else:
        generate_kwargs["do_sample"] = False

    outputs = model.generate(**inputs, **generate_kwargs)

    # Decode only the new tokens (strip the input prompt)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()


def run_inference(
    model,
    tokenizer,
    samples: List[Dict],
    task: str,
    model_name: str = "",
    max_new_tokens: int = 64,
    use_fewshot: bool = True,
    save_path: Optional[str] = None,
) -> List[Dict]:
    """
    Run inference on a batch of samples for a given task.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        samples: List of normalized samples.
        task: Task type.
        model_name: Name of the model (for logging).
        max_new_tokens: Maximum tokens to generate per sample.
        use_fewshot: Whether to use few-shot prompting.
        save_path: Optional path to save results as JSONL.

    Returns:
        List of result dicts with predictions.
    """
    results = []
    total_time = 0

    print(f"\n🔮 Running inference — {model_name} on {task}")
    print(f"   Samples: {len(samples)}, Max tokens: {max_new_tokens}")
    print("-" * 50)

    for sample in tqdm(samples, desc=f"{model_name}/{task}"):
        prompt = build_prompt(sample, task, use_fewshot=use_fewshot)

        start_time = time.time()
        prediction = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - start_time
        total_time += elapsed

        result = {
            "id": sample["id"],
            "task": task,
            "language": sample["language"],
            "model": model_name,
            "question": sample["question"],
            "gold_answer": sample["answer"],
            "prediction": prediction,
            "inference_time_s": round(elapsed, 3),
        }

        # Include context/options for reference
        if sample.get("context"):
            result["context"] = sample["context"][:200] + "..."
        if sample.get("options"):
            result["options"] = sample["options"]

        results.append(result)

    # Summary stats
    avg_time = total_time / max(len(samples), 1)
    print(f"\n   ⏱️  Total time: {total_time:.1f}s, Avg: {avg_time:.2f}s/sample")

    # Optionally save results
    if save_path:
        _save_results(results, save_path)

    return results


def _save_results(results: List[Dict], save_path: str):
    """Save results as JSONL file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"   💾 Results saved to {save_path}")


def load_results(path: str) -> List[Dict]:
    """Load results from a JSONL file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


# ─── Full Benchmark Run ──────────────────────────────────────────────────────

def run_full_benchmark(
    model,
    tokenizer,
    model_key: str,
    model_name: str,
    datasets: Dict[str, List[Dict]],
    results_dir: str = "results/raw",
    use_fewshot: bool = True,
) -> Dict[str, List[Dict]]:
    """
    Run full benchmark for a single model across all tasks.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        model_key: Model key (e.g., "sarvam-2b").
        model_name: Human-readable model name.
        datasets: Dict mapping task names to samples.
        results_dir: Directory to save raw results.
        use_fewshot: Whether to use few-shot prompting.

    Returns:
        Dictionary mapping task names to result lists.
    """
    all_results = {}

    for task_name, samples in datasets.items():
        if not samples:
            print(f"⚠️  No samples for {task_name}. Skipping.")
            continue

        save_path = os.path.join(results_dir, f"{model_key}_{task_name}.jsonl")

        # Adjust max_new_tokens based on task
        max_tokens = 64
        if task_name == "math_reasoning":
            max_tokens = 16  # Just need the option letter
        elif task_name == "code_mixed_qa":
            max_tokens = 48

        results = run_inference(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            task=task_name,
            model_name=model_name,
            max_new_tokens=max_tokens,
            use_fewshot=use_fewshot,
            save_path=save_path,
        )

        all_results[task_name] = results

    return all_results


if __name__ == "__main__":
    # Test prompt construction
    test_sample = {
        "id": "test_1",
        "question": "भारत की राजधानी क्या है?",
        "context": "भारत दक्षिण एशिया में स्थित एक देश है। इसकी राजधानी नई दिल्ली है।",
        "answer": "नई दिल्ली",
        "language": "hindi",
        "task": "reading_comprehension",
        "options": None,
        "correct_option": None,
    }

    print("=== Reading Comprehension Prompt ===")
    print(build_prompt(test_sample, "reading_comprehension"))
    print("\n=== Math Reasoning Prompt ===")
    test_math = {**test_sample, "options": ["A. 3", "B. 4", "C. 5"], "question": "2+2=?"}
    print(build_prompt(test_math, "math_reasoning"))
    print("\n=== Code-Mixed Prompt ===")
    test_cm = {**test_sample, "question": "India ka capital kya hai?"}
    print(build_prompt(test_cm, "code_mixed_qa"))
