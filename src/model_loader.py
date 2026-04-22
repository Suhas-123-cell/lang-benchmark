"""
Model Loader Module
===================
Handles loading and configuration of the three benchmark models:
1. Sarvam-2B (sarvamai/sarvam-2b-v0.5) — Indic-focused 2B param model
2. Gemma-2B (google/gemma-2b) — Google's efficient 2B param model
3. Llama-3.2-1B (meta-llama/Llama-3.2-1B) — Meta's compact 1B param model

All models are loaded with 4-bit quantization for Colab T4 compatibility.
"""

import gc
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ─── Model Registry ──────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Configuration for a benchmark model."""
    name: str
    hf_id: str
    description: str
    model_type: str  # "sarvam", "gemma", "llama"
    max_context: int = 2048
    trust_remote_code: bool = False
    extra_kwargs: Dict = field(default_factory=dict)


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "sarvam-2b": ModelConfig(
        name="Sarvam-2B",
        hf_id="sarvamai/sarvam-2b-v0.5",
        description="Indic-focused 2B model pre-trained on 2T tokens (10 Indic languages + English)",
        model_type="sarvam",
        max_context=2048,
        trust_remote_code=True,
    ),
    "gemma-2b": ModelConfig(
        name="Gemma-2B",
        hf_id="google/gemma-2b",
        description="Google's efficient 2B parameter model with strong reasoning capabilities",
        model_type="gemma",
        max_context=8192,
        trust_remote_code=False,
    ),
    "llama-3.2-1b": ModelConfig(
        name="Llama-3.2-1B",
        hf_id="meta-llama/Llama-3.2-1B",
        description="Meta's compact 1B parameter model with multilingual support",
        model_type="llama",
        max_context=8192,  # Supports 128k but we cap for memory
        trust_remote_code=False,
    ),
}


# ─── Quantization Config ─────────────────────────────────────────────────────

def get_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "float16",
) -> Optional[BitsAndBytesConfig]:
    """
    Get BitsAndBytes quantization config for memory-efficient loading.

    4-bit quantization reduces VRAM usage by ~4x, allowing 2B models
    to fit on a T4 GPU (16GB VRAM) with room for inference.
    """
    if not load_in_4bit:
        return None

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype, torch.float16)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,  # Nested quantization for extra savings
    )


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_key: str,
    quantize: bool = True,
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
) -> Tuple:
    """
    Load a model and its tokenizer from the registry.

    Args:
        model_key: Key from MODEL_REGISTRY (e.g., "sarvam-2b").
        quantize: Whether to apply 4-bit quantization.
        device_map: Device placement strategy ("auto" for GPU).
        cache_dir: Optional cache directory for model weights.

    Returns:
        Tuple of (model, tokenizer, config).
    """
    if model_key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model_key}'. Available: {available}"
        )

    config = MODEL_REGISTRY[model_key]
    print(f"\n{'=' * 60}")
    print(f"🤖 Loading {config.name}")
    print(f"   HuggingFace ID: {config.hf_id}")
    print(f"   Quantization: {'4-bit NF4' if quantize else 'Full precision'}")
    print(f"{'=' * 60}")

    # Quantization config
    quant_config = get_quantization_config() if quantize else None

    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_id,
        trust_remote_code=config.trust_remote_code,
        cache_dir=cache_dir,
    )

    # Ensure pad token is set (many base models don't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    print("🧠 Loading model weights...")
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "device_map": device_map,
        "torch_dtype": torch.float16,
        "cache_dir": cache_dir,
        **config.extra_kwargs,
    }

    if quant_config:
        model_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(
        config.hf_id,
        **model_kwargs,
    )

    model.eval()  # Set to evaluation mode

    # Report memory usage
    _report_memory(config.name)

    print(f"✅ {config.name} loaded successfully!")
    return model, tokenizer, config


def _report_memory(model_name: str):
    """Report GPU memory usage after loading a model."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   📊 GPU Memory — Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"   📊 Running on Apple MPS (Metal Performance Shaders)")
    else:
        print(f"   📊 Running on CPU (inference will be slow)")


def unload_model(model, tokenizer):
    """
    Unload a model to free GPU memory before loading the next one.

    On Colab free tier, we can typically only hold one 2B model
    in memory at a time with 4-bit quantization.
    """
    print("🗑️  Unloading model to free memory...")
    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("   ✅ Memory freed")


# ─── Tokenizer Analysis ──────────────────────────────────────────────────────

def compute_fertility(
    tokenizer,
    text: str,
    model_name: str = "",
) -> Dict:
    """
    Compute tokenizer fertility score for a given text.

    Fertility = number_of_tokens / number_of_words

    A lower fertility score means the tokenizer is more efficient
    for that language. Sarvam-2B's tokenizer is specifically designed
    to have low fertility (~2) for Indic languages.

    Args:
        tokenizer: The tokenizer to analyze.
        text: Input text to tokenize.
        model_name: Name of the model for reporting.

    Returns:
        Dictionary with fertility metrics.
    """
    words = text.split()
    tokens = tokenizer.encode(text, add_special_tokens=False)

    num_words = len(words)
    num_tokens = len(tokens)
    fertility = num_tokens / max(num_words, 1)

    return {
        "model": model_name,
        "num_words": num_words,
        "num_tokens": num_tokens,
        "fertility": round(fertility, 3),
    }


def compare_tokenizer_fertility(
    tokenizers: Dict[str, object],
    test_texts: Dict[str, str],
) -> "pd.DataFrame":
    """
    Compare tokenizer fertility across models and languages.

    Args:
        tokenizers: Dict mapping model names to tokenizer objects.
        test_texts: Dict mapping language names to sample texts.

    Returns:
        DataFrame with fertility scores for each model × language combination.
    """
    import pandas as pd

    results = []
    for model_name, tokenizer in tokenizers.items():
        for lang, text in test_texts.items():
            fertility_data = compute_fertility(tokenizer, text, model_name)
            fertility_data["language"] = lang
            results.append(fertility_data)

    df = pd.DataFrame(results)
    return df


# ─── Convenience ──────────────────────────────────────────────────────────────

def load_all_models(
    model_keys: list = None,
    quantize: bool = True,
    cache_dir: Optional[str] = None,
) -> Dict[str, Tuple]:
    """
    Load all benchmark models sequentially.

    WARNING: On free-tier Colab, loading all models simultaneously
    may exceed memory. Consider loading one at a time with unload_model().

    Args:
        model_keys: List of model keys to load. Defaults to all.
        quantize: Whether to apply 4-bit quantization.
        cache_dir: Optional cache directory.

    Returns:
        Dictionary mapping model keys to (model, tokenizer, config) tuples.
    """
    if model_keys is None:
        model_keys = list(MODEL_REGISTRY.keys())

    models = {}
    for key in model_keys:
        model, tokenizer, config = load_model_and_tokenizer(
            key, quantize=quantize, cache_dir=cache_dir
        )
        models[key] = (model, tokenizer, config)

    return models


def get_model_info() -> str:
    """Get a formatted summary of all registered models."""
    lines = ["📋 Registered Models:", "=" * 50]
    for key, config in MODEL_REGISTRY.items():
        lines.append(f"\n  {key}:")
        lines.append(f"    Name: {config.name}")
        lines.append(f"    HF ID: {config.hf_id}")
        lines.append(f"    Description: {config.description}")
        lines.append(f"    Max Context: {config.max_context}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_model_info())
