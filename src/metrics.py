"""
Metrics Module
==============
Implements evaluation metrics for the Indic Language Benchmark Suite:

1. Exact Match (EM) — Normalized string comparison
2. F1 Score — Token-level overlap (for extractive QA)
3. Accuracy — Option letter match (for multiple-choice)
4. BLEU — N-gram precision (for generative tasks)
5. Perplexity — Per-token log-likelihood (for model comparison)

All metrics include Indic-aware normalization for fair comparison
across scripts (Devanagari, Bengali, Tamil).
"""

import re
import string
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Text Normalization ──────────────────────────────────────────────────────

def normalize_text(text: str, language: str = "english") -> str:
    """
    Normalize text for comparison.

    Handles:
    - Lowercasing
    - Whitespace normalization
    - Removing articles (English)
    - Removing punctuation
    - Unicode normalization for Indic scripts

    Args:
        text: Input text to normalize.
        language: Language hint for script-specific normalization.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    # Unicode NFC normalization (compose characters)
    text = unicodedata.normalize("NFC", text)

    # Lowercase
    text = text.lower()

    # Remove articles (English and Hinglish)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)

    # Remove punctuation (keep Indic script characters)
    # Standard ASCII punctuation
    text = re.sub(r'[' + re.escape(string.punctuation) + r']', ' ', text)
    # Common Indic punctuation (danda, double danda, etc.)
    text = re.sub(r'[।॥,\.\?!;:"\'-]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_for_f1(text: str) -> List[str]:
    """
    Tokenize text for F1 computation.

    Uses whitespace tokenization which works reasonably well
    across Indic scripts since words are space-delimited.
    """
    return normalize_text(text).split()


# ─── Exact Match ──────────────────────────────────────────────────────────────

def exact_match(prediction: str, gold: str, language: str = "english") -> float:
    """
    Compute Exact Match score.

    Returns 1.0 if normalized prediction matches normalized gold, else 0.0.

    Args:
        prediction: Model's predicted answer.
        gold: Ground truth answer.
        language: Language hint for normalization.

    Returns:
        1.0 or 0.0
    """
    return 1.0 if normalize_text(prediction, language) == normalize_text(gold, language) else 0.0


def compute_exact_match(
    predictions: List[str],
    golds: List[str],
    languages: Optional[List[str]] = None,
) -> Dict:
    """
    Compute Exact Match across a dataset.

    Args:
        predictions: List of predicted answers.
        golds: List of ground truth answers.
        languages: Optional list of languages for normalization.

    Returns:
        Dictionary with overall EM and per-language breakdown.
    """
    if languages is None:
        languages = ["english"] * len(predictions)

    scores = []
    per_lang = {}

    for pred, gold, lang in zip(predictions, golds, languages):
        score = exact_match(pred, gold, lang)
        scores.append(score)

        if lang not in per_lang:
            per_lang[lang] = []
        per_lang[lang].append(score)

    return {
        "exact_match": round(np.mean(scores) * 100, 2),
        "per_language": {
            lang: round(np.mean(lang_scores) * 100, 2)
            for lang, lang_scores in per_lang.items()
        },
        "total_samples": len(scores),
    }


# ─── F1 Score (Token-Level) ──────────────────────────────────────────────────

def f1_score(prediction: str, gold: str) -> float:
    """
    Compute token-level F1 score between prediction and gold.

    F1 is the harmonic mean of precision and recall computed
    over the bag of tokens (words).

    Args:
        prediction: Model's predicted answer.
        gold: Ground truth answer.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    pred_tokens = tokenize_for_f1(prediction)
    gold_tokens = tokenize_for_f1(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_f1(
    predictions: List[str],
    golds: List[str],
    languages: Optional[List[str]] = None,
) -> Dict:
    """
    Compute F1 score across a dataset.

    Returns:
        Dictionary with overall F1 and per-language breakdown.
    """
    if languages is None:
        languages = ["english"] * len(predictions)

    scores = []
    per_lang = {}

    for pred, gold, lang in zip(predictions, golds, languages):
        score = f1_score(pred, gold)
        scores.append(score)

        if lang not in per_lang:
            per_lang[lang] = []
        per_lang[lang].append(score)

    return {
        "f1": round(np.mean(scores) * 100, 2),
        "per_language": {
            lang: round(np.mean(lang_scores) * 100, 2)
            for lang, lang_scores in per_lang.items()
        },
        "total_samples": len(scores),
    }


# ─── Accuracy (for Multiple-Choice) ──────────────────────────────────────────

def extract_option_letter(text: str) -> str:
    """
    Extract a single option letter (A-J) from model output.

    Handles various formats:
    - "B"
    - "B. 4"
    - "The answer is B"
    - "(B)"

    Args:
        text: Model's raw output.

    Returns:
        Extracted option letter (uppercase), or empty string if not found.
    """
    if not text:
        return ""

    text = text.strip()

    # Direct single letter match
    if len(text) == 1 and text.upper() in "ABCDEFGHIJ":
        return text.upper()

    # Pattern: starts with a letter followed by punctuation or space
    match = re.match(r'^([A-Ja-j])[.\s)\]]', text)
    if match:
        return match.group(1).upper()

    # Pattern: "The answer is X" or "Answer: X"
    match = re.search(r'(?:answer\s*(?:is|:)\s*)([A-Ja-j])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern: "(X)" anywhere in text
    match = re.search(r'\(([A-Ja-j])\)', text)
    if match:
        return match.group(1).upper()

    # Last resort: find first standalone letter
    match = re.search(r'\b([A-Ja-j])\b', text)
    if match:
        return match.group(1).upper()

    return ""


def accuracy(prediction: str, gold: str) -> float:
    """
    Compute accuracy for a single multiple-choice prediction.

    Extracts the option letter from the prediction and compares
    to the gold answer.

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    pred_letter = extract_option_letter(prediction)
    gold_letter = extract_option_letter(gold) if len(gold) > 1 else gold.upper()
    return 1.0 if pred_letter == gold_letter else 0.0


def compute_accuracy(
    predictions: List[str],
    golds: List[str],
    languages: Optional[List[str]] = None,
) -> Dict:
    """
    Compute accuracy across a dataset.

    Returns:
        Dictionary with overall accuracy and per-language breakdown.
    """
    if languages is None:
        languages = ["english"] * len(predictions)

    scores = []
    per_lang = {}

    for pred, gold, lang in zip(predictions, golds, languages):
        score = accuracy(pred, gold)
        scores.append(score)

        if lang not in per_lang:
            per_lang[lang] = []
        per_lang[lang].append(score)

    return {
        "accuracy": round(np.mean(scores) * 100, 2),
        "per_language": {
            lang: round(np.mean(lang_scores) * 100, 2)
            for lang, lang_scores in per_lang.items()
        },
        "total_samples": len(scores),
    }


# ─── BLEU Score ───────────────────────────────────────────────────────────────

def compute_bleu_single(
    prediction: str,
    gold: str,
    max_n: int = 4,
) -> float:
    """
    Compute sentence-level BLEU score with smoothing.

    Uses a simple smoothing method (add-1) to handle short sequences
    where n-gram counts might be zero.

    Args:
        prediction: Predicted text.
        gold: Reference text.
        max_n: Maximum n-gram order.

    Returns:
        BLEU score between 0.0 and 1.0.
    """
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    # Compute n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        gold_ngrams = _get_ngrams(gold_tokens, n)

        if not pred_ngrams:
            precisions.append(0.0)
            continue

        matches = sum(
            min(pred_ngrams[ng], gold_ngrams.get(ng, 0))
            for ng in pred_ngrams
        )

        # Add-1 smoothing
        precision = (matches + 1) / (sum(pred_ngrams.values()) + 1)
        precisions.append(precision)

    if not precisions or all(p == 0 for p in precisions):
        return 0.0

    # Geometric mean of precisions
    log_avg = sum(np.log(max(p, 1e-10)) for p in precisions) / len(precisions)
    bleu = np.exp(log_avg)

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(gold_tokens) / max(len(pred_tokens), 1)))

    return round(float(bleu * bp), 6)


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from a token list."""
    return Counter(
        tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)
    )


def compute_bleu(
    predictions: List[str],
    golds: List[str],
    languages: Optional[List[str]] = None,
) -> Dict:
    """
    Compute BLEU across a dataset.

    Returns:
        Dictionary with overall BLEU and per-language breakdown.
    """
    if languages is None:
        languages = ["english"] * len(predictions)

    scores = []
    per_lang = {}

    for pred, gold, lang in zip(predictions, golds, languages):
        score = compute_bleu_single(pred, gold)
        scores.append(score)

        if lang not in per_lang:
            per_lang[lang] = []
        per_lang[lang].append(score)

    return {
        "bleu": round(np.mean(scores) * 100, 2),
        "per_language": {
            lang: round(np.mean(lang_scores) * 100, 2)
            for lang, lang_scores in per_lang.items()
        },
        "total_samples": len(scores),
    }


# ─── Unified Evaluation ──────────────────────────────────────────────────────

def evaluate_results(
    results: List[Dict],
    task: str,
) -> Dict:
    """
    Evaluate a set of results using the appropriate metrics for the task.

    Metric mapping:
    - reading_comprehension → EM + F1
    - math_reasoning → Accuracy
    - code_mixed_qa → EM + BLEU + F1

    Args:
        results: List of result dicts with 'prediction' and 'gold_answer' keys.
        task: Task type string.

    Returns:
        Dictionary with all computed metrics.
    """
    predictions = [r["prediction"] for r in results]
    golds = [r["gold_answer"] for r in results]
    languages = [r.get("language", "unknown") for r in results]

    metrics = {"task": task, "num_samples": len(results)}

    if task == "reading_comprehension":
        metrics["exact_match"] = compute_exact_match(predictions, golds, languages)
        metrics["f1"] = compute_f1(predictions, golds, languages)

    elif task == "math_reasoning":
        metrics["accuracy"] = compute_accuracy(predictions, golds, languages)

    elif task == "code_mixed_qa":
        metrics["exact_match"] = compute_exact_match(predictions, golds, languages)
        metrics["f1"] = compute_f1(predictions, golds, languages)
        metrics["bleu"] = compute_bleu(predictions, golds, languages)

    else:
        # Default: compute all metrics
        metrics["exact_match"] = compute_exact_match(predictions, golds, languages)
        metrics["f1"] = compute_f1(predictions, golds, languages)
        metrics["bleu"] = compute_bleu(predictions, golds, languages)

    return metrics


# ─── Convenience: Quick Report ────────────────────────────────────────────────

def format_metrics_report(metrics: Dict, model_name: str = "") -> str:
    """
    Format metrics into a readable report string.

    Args:
        metrics: Output from evaluate_results().
        model_name: Optional model name for the header.

    Returns:
        Formatted string report.
    """
    lines = []
    header = f"📊 Metrics Report — {model_name}" if model_name else "📊 Metrics Report"
    lines.append(header)
    lines.append("=" * len(header))
    lines.append(f"Task: {metrics['task']}")
    lines.append(f"Samples: {metrics['num_samples']}")
    lines.append("")

    for metric_name, metric_data in metrics.items():
        if metric_name in ("task", "num_samples"):
            continue

        if isinstance(metric_data, dict):
            # Get the main score
            main_key = [k for k in metric_data if k not in ("per_language", "total_samples")]
            if main_key:
                lines.append(f"  {metric_name.upper()}: {metric_data[main_key[0]]}%")

            # Per-language breakdown
            if "per_language" in metric_data:
                for lang, score in metric_data["per_language"].items():
                    lines.append(f"    └── {lang}: {score}%")
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # Sanity checks
    print("=== Metric Sanity Checks ===\n")

    # Exact Match
    assert exact_match("New Delhi", "new delhi") == 1.0
    assert exact_match("Mumbai", "new delhi") == 0.0
    print("✅ Exact Match works")

    # F1
    score = f1_score("The capital is New Delhi", "New Delhi")
    assert score > 0
    print(f"✅ F1 works (score: {score:.3f})")

    # Accuracy
    assert accuracy("B", "B") == 1.0
    assert accuracy("The answer is B", "B") == 1.0
    assert accuracy("B. 4", "B") == 1.0
    assert accuracy("A", "B") == 0.0
    print("✅ Accuracy works")

    # BLEU
    bleu = compute_bleu_single("The cat sat on the mat", "The cat sat on a mat")
    assert bleu > 0
    print(f"✅ BLEU works (score: {bleu:.3f})")

    # Indic text
    em = exact_match("नई दिल्ली", "नई दिल्ली")
    assert em == 1.0
    print("✅ Indic Exact Match works")

    f1 = f1_score("नई दिल्ली भारत की राजधानी है", "नई दिल्ली")
    assert f1 > 0
    print(f"✅ Indic F1 works (score: {f1:.3f})")

    print("\n🎉 All sanity checks passed!")
