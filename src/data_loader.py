"""
Data Loader Module
==================
Handles loading and preprocessing of all three benchmark datasets:
1. IndicQA (AI4Bharat) — Reading Comprehension
2. IndicMMLU-Pro (LinguaLift) — Math/STEM Reasoning
3. Code-Mixed QA — Hand-crafted Hinglish samples

All datasets are normalized to a common schema for consistent evaluation.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
import pandas as pd


# ─── Common Schema ───────────────────────────────────────────────────────────

def normalize_sample(
    question: str,
    answer: str,
    language: str,
    task: str,
    context: Optional[str] = None,
    options: Optional[List[str]] = None,
    correct_option: Optional[str] = None,
    sample_id: Optional[str] = None,
) -> Dict:
    """Normalize a sample to a common benchmark schema."""
    return {
        "id": sample_id,
        "question": question.strip(),
        "context": context.strip() if context else None,
        "options": options,
        "correct_option": correct_option,
        "answer": answer.strip(),
        "language": language,
        "task": task,
    }


# ─── Task 1: IndicQA (Reading Comprehension) ─────────────────────────────────

# Unicode script detection for language filtering
import unicodedata

def _detect_indic_language(text: str) -> str:
    """Detect the Indic language of a text by its Unicode script."""
    for char in text[:300]:
        name = unicodedata.name(char, "")
        if "DEVANAGARI" in name:
            return "hindi"
        if "BENGALI" in name:
            return "bengali"
        if "TAMIL" in name:
            return "tamil"
        if "TELUGU" in name:
            return "telugu"
        if "KANNADA" in name:
            return "kannada"
        if "MALAYALAM" in name:
            return "malayalam"
        if "GUJARATI" in name:
            return "gujarati"
        if "GURMUKHI" in name:
            return "punjabi"
        if "ORIYA" in name:
            return "oriya"
    return "unknown"


def load_indicqa(
    languages: List[str] = ["hindi", "bengali", "tamil"],
    max_samples_per_lang: int = 500,
    split: str = "test",
) -> List[Dict]:
    """
    Load IndicQA dataset from AI4Bharat.

    IndicQA is an extractive QA dataset based on Wikipedia passages
    in 11 Indic languages. Each sample has a context paragraph,
    a question, and an extractive answer span.

    Uses parquet revision since the original loading script is
    deprecated in newer versions of the `datasets` library.
    Languages are detected via Unicode script analysis.

    Args:
        languages: List of languages to load (hindi, bengali, tamil).
        max_samples_per_lang: Maximum number of samples per language.
        split: Dataset split to use (typically 'test').

    Returns:
        List of normalized sample dicts.
    """
    all_samples = []

    print(f"📖 Loading IndicQA dataset (parquet)...")
    try:
        dataset = load_dataset(
            "ai4bharat/IndicQA",
            revision="refs/convert/parquet",
            split=split,
        )
    except Exception as e:
        print(f"❌ Failed to load IndicQA: {e}")
        return all_samples

    print(f"   📦 Total samples in dataset: {len(dataset)}")

    # Group by detected language
    lang_counts = {lang: 0 for lang in languages}

    for i, sample in enumerate(dataset):
        # Detect language from context text
        context = sample.get("context", "")
        detected_lang = _detect_indic_language(context)

        if detected_lang not in languages:
            continue

        if lang_counts[detected_lang] >= max_samples_per_lang:
            continue

        # IndicQA follows SQuAD-style format
        answers = sample.get("answers", {})
        answer_texts = answers.get("text", [])

        # Skip samples without valid answers
        if not answer_texts or not answer_texts[0]:
            continue

        normalized = normalize_sample(
            question=sample["question"],
            answer=answer_texts[0],  # Take the first answer
            language=detected_lang,
            task="reading_comprehension",
            context=context,
            sample_id=f"indicqa_{detected_lang}_{lang_counts[detected_lang]}",
        )
        all_samples.append(normalized)
        lang_counts[detected_lang] += 1

        # Check if we have enough for all languages
        if all(lang_counts[lang] >= max_samples_per_lang for lang in languages):
            break

    for lang in languages:
        print(f"   ✅ {lang}: {lang_counts[lang]} samples")

    print(f"\n📊 Total IndicQA samples: {len(all_samples)}")
    return all_samples


# ─── Task 2: IndicMMLU-Pro (Math/STEM Reasoning) ─────────────────────────────

INDICMMLU_LANG_MAP = {
    "hindi": "hindi",
    "bengali": "bengali",
    "tamil": "tamil",
}

# STEM-related categories to filter for math reasoning
STEM_CATEGORIES = [
    "math", "mathematics", "physics", "chemistry", "biology",
    "computer_science", "engineering", "statistics",
    "abstract_algebra", "college_mathematics", "elementary_mathematics",
    "high_school_mathematics", "high_school_statistics",
    "college_physics", "high_school_physics",
    "college_chemistry", "high_school_chemistry",
    "college_biology", "high_school_biology",
    "college_computer_science", "high_school_computer_science",
    "electrical_engineering", "machine_learning",
]


def load_indicmmlu(
    languages: List[str] = ["hindi", "bengali", "tamil"],
    max_samples_per_lang: int = 200,
    stem_only: bool = True,
) -> List[Dict]:
    """
    Load IndicMMLU-Pro dataset for math/STEM evaluation.

    IndicMMLU-Pro is a multilingual adaptation of MMLU-Pro with
    multiple-choice questions across many subjects in Indic languages.

    Args:
        languages: List of languages to load.
        max_samples_per_lang: Maximum samples per language.
        stem_only: If True, only load STEM-related categories.

    Returns:
        List of normalized sample dicts.
    """
    all_samples = []

    for lang in languages:
        lang_code = INDICMMLU_LANG_MAP.get(lang)
        if not lang_code:
            print(f"⚠️  Language '{lang}' not found in IndicMMLU-Pro. Skipping.")
            continue

        print(f"🔢 Loading IndicMMLU-Pro — {lang} ({lang_code})...")

        try:
            # Load with full language name as config
            dataset = load_dataset(
                "LinguaLift/IndicMMLU-Pro",
                lang_code,
                split="test",
            )
        except Exception:
            try:
                # Fallback: load default and filter
                dataset = load_dataset(
                    "LinguaLift/IndicMMLU-Pro",
                    split="test",
                )
                dataset = dataset.filter(
                    lambda x: x.get("language", "") == lang_code
                )
            except Exception as e:
                print(f"❌ Failed to load IndicMMLU-Pro for {lang}: {e}")
                continue

        count = 0
        for i, sample in enumerate(dataset):
            if count >= max_samples_per_lang:
                break

            # Filter for STEM categories if requested
            category = sample.get("category", "").lower().replace(" ", "_")
            if stem_only and category not in STEM_CATEGORIES:
                continue

            # Extract options (typically labeled A through J in MMLU-Pro)
            options = sample.get("options", [])
            if isinstance(options, str):
                try:
                    options = json.loads(options)
                except json.JSONDecodeError:
                    options = [o.strip() for o in options.split(",")]

            # Get the correct answer
            answer = sample.get("answer", "")
            answer_index = sample.get("answer_index", None)

            # Build option labels
            option_labels = [chr(65 + j) for j in range(len(options))]
            options_formatted = [
                f"{label}. {opt}" for label, opt in zip(option_labels, options)
            ]

            # Determine correct option letter
            if answer_index is not None and isinstance(answer_index, int):
                correct_option = chr(65 + answer_index)
            elif answer in option_labels:
                correct_option = answer
            else:
                correct_option = answer

            normalized = normalize_sample(
                question=sample.get("question", ""),
                answer=correct_option,
                language=lang,
                task="math_reasoning",
                options=options_formatted,
                correct_option=correct_option,
                sample_id=f"indicmmlu_{lang}_{i}",
            )
            all_samples.append(normalized)
            count += 1

        print(f"   ✅ Loaded {count} samples for {lang}")

    # If STEM-only yielded too few samples, relax the filter
    if stem_only and len(all_samples) < 50:
        print("⚠️  Too few STEM samples found. Reloading without STEM filter...")
        return load_indicmmlu(
            languages=languages,
            max_samples_per_lang=max_samples_per_lang,
            stem_only=False,
        )

    print(f"\n📊 Total IndicMMLU-Pro samples: {len(all_samples)}")
    return all_samples


# ─── Task 3: Code-Mixed QA (Hinglish) ────────────────────────────────────────

def load_code_mixed_qa(
    data_path: str = "data/code_mixed_qa.json",
) -> List[Dict]:
    """
    Load hand-crafted Hinglish code-mixed QA samples.

    Args:
        data_path: Path to the JSON file containing code-mixed samples.

    Returns:
        List of normalized sample dicts.
    """
    # Resolve path relative to project root
    if not os.path.isabs(data_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, data_path)

    print(f"🗣️  Loading Code-Mixed QA from {data_path}...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Code-mixed QA file not found at {data_path}. "
            "Please create it first using the provided template."
        )

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    all_samples = []
    for sample in raw_data:
        normalized = normalize_sample(
            question=sample["question"],
            answer=sample["answer"],
            language="hinglish",
            task="code_mixed_qa",
            sample_id=f"codemixed_{sample.get('id', len(all_samples))}",
        )
        # Preserve extra metadata
        normalized["category"] = sample.get("category", "general")
        all_samples.append(normalized)

    print(f"   ✅ Loaded {len(all_samples)} code-mixed samples")
    return all_samples


# ─── Unified Loader ──────────────────────────────────────────────────────────

def load_all_datasets(
    languages: List[str] = ["hindi", "bengali", "tamil"],
    max_indicqa: int = 500,
    max_indicmmlu: int = 200,
    code_mixed_path: str = "data/code_mixed_qa.json",
) -> Dict[str, List[Dict]]:
    """
    Load all three benchmark datasets.

    Returns:
        Dictionary mapping task names to lists of normalized samples.
    """
    print("=" * 60)
    print("🚀 Loading Indic Language Benchmark Datasets")
    print("=" * 60)

    datasets_loaded = {}

    # Task 1: Reading Comprehension
    print("\n─── Task 1: Reading Comprehension (IndicQA) ───")
    datasets_loaded["reading_comprehension"] = load_indicqa(
        languages=languages,
        max_samples_per_lang=max_indicqa,
    )

    # Task 2: Math Reasoning
    print("\n─── Task 2: Math Reasoning (IndicMMLU-Pro) ───")
    datasets_loaded["math_reasoning"] = load_indicmmlu(
        languages=languages,
        max_samples_per_lang=max_indicmmlu,
    )

    # Task 3: Code-Mixed QA
    print("\n─── Task 3: Code-Mixed QA (Hinglish) ───")
    try:
        datasets_loaded["code_mixed_qa"] = load_code_mixed_qa(
            data_path=code_mixed_path,
        )
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        datasets_loaded["code_mixed_qa"] = []

    # Summary
    print("\n" + "=" * 60)
    print("📊 Dataset Loading Summary:")
    for task, samples in datasets_loaded.items():
        lang_counts = {}
        for s in samples:
            lang_counts[s["language"]] = lang_counts.get(s["language"], 0) + 1
        print(f"   {task}: {len(samples)} total — {lang_counts}")
    print("=" * 60)

    return datasets_loaded


def dataset_to_dataframe(samples: List[Dict]) -> pd.DataFrame:
    """Convert a list of normalized samples to a pandas DataFrame."""
    return pd.DataFrame(samples)


if __name__ == "__main__":
    # Quick test — load all datasets
    data = load_all_datasets(max_indicqa=10, max_indicmmlu=10)
    for task, samples in data.items():
        print(f"\n{task}: {len(samples)} samples")
        if samples:
            print(f"  Example: {samples[0]}")
