"""
Model Loader Module
===================
Handles loading and configuration of the three benchmark models:
1. Sarvam-2B (sarvamai/sarvam-2b-v0.5) — Indic-focused 2B param model
2. Gemma-2B (google/gemma-2b) — Google's efficient 2B param model
3. Llama-3.2-1B (meta-llama/Llama-3.2-1B) — Meta's compact 1B param model

Supports:
- CUDA GPUs with optional 4-bit quantization (bitsandbytes)
- Apple Silicon MPS with float16/float32
- CPU fallback
"""

import gc
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Device Detection ────────────────────────────────────────────────────────

def get_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_torch_dtype(device: str) -> torch.dtype:
    """Get the optimal dtype for the device."""
    if device == "cuda":
        return torch.float16
    elif device == "mps":
        # MPS supports float16 for most ops in recent PyTorch
        return torch.float16
    return torch.float32


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


# ─── Quantization Config (CUDA only) ─────────────────────────────────────────

def get_quantization_config(device: str):
    """
    Get BitsAndBytes quantization config (CUDA only).

    4-bit quantization reduces VRAM usage by ~4x, allowing 2B models
    to fit on a T4 GPU (16GB VRAM). Not supported on MPS or CPU.
    """
    if device != "cuda":
        return None

    try:
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    except ImportError:
        print("⚠️  bitsandbytes not installed. Loading in full precision.")
        return None


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_key: str,
    quantize: bool = True,
    device_map: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple:
    """
    Load a model and its tokenizer from the registry.

    Automatically detects CUDA/MPS/CPU and configures accordingly:
    - CUDA: Uses 4-bit quantization if quantize=True
    - MPS: Uses float16, loads to MPS device
    - CPU: Uses float32 (slow but works)

    Args:
        model_key: Key from MODEL_REGISTRY (e.g., "sarvam-2b").
        quantize: Whether to apply 4-bit quantization (CUDA only).
        device_map: Device placement. Auto-detected if None.
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
    device = get_device()
    dtype = get_torch_dtype(device)

    print(f"\n{'=' * 60}")
    print(f"🤖 Loading {config.name}")
    print(f"   HuggingFace ID: {config.hf_id}")
    print(f"   Device: {device.upper()}")
    print(f"   Dtype: {dtype}")
    if device == "cuda" and quantize:
        print(f"   Quantization: 4-bit NF4")
    else:
        print(f"   Quantization: None (native {dtype})")
    print(f"{'=' * 60}")

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

    # Build model kwargs based on device
    print("🧠 Loading model weights...")
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": dtype,
        "cache_dir": cache_dir,
        "low_cpu_mem_usage": True,
        **config.extra_kwargs,
    }

    if device == "cuda":
        model_kwargs["device_map"] = device_map or "auto"
        if quantize:
            quant_config = get_quantization_config(device)
            if quant_config:
                model_kwargs["quantization_config"] = quant_config
    elif device == "mps":
        # For MPS: load to CPU first, then move to MPS
        # device_map="auto" doesn't work well with MPS
        model_kwargs["device_map"] = None
    else:
        model_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(
        config.hf_id,
        **model_kwargs,
    )

    # Move to MPS if available
    if device == "mps":
        print("   🍎 Moving model to MPS...")
        model = model.to("mps")

    model.eval()  # Set to evaluation mode

    # Report memory usage
    _report_memory(config.name, device)

    print(f"✅ {config.name} loaded successfully!")
    return model, tokenizer, config


def _report_memory(model_name: str, device: str = None):
    """Report GPU memory usage after loading a model."""
    if device is None:
        device = get_device()

    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   📊 GPU Memory — Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    elif device == "mps":
        # MPS memory reporting (available in recent PyTorch)
        try:
            allocated = torch.mps.current_allocated_memory() / 1024**3
            print(f"   📊 MPS Memory — Allocated: {allocated:.2f} GB")
        except AttributeError:
            print(f"   📊 Running on Apple MPS (Metal Performance Shaders)")
    else:
        print(f"   📊 Running on CPU (inference will be slow)")


def unload_model(model, tokenizer):
    """
    Unload a model to free GPU memory before loading the next one.

    Handles CUDA, MPS, and CPU cleanup.
    """
    print("🗑️  Unloading model to free memory...")
    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

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

    WARNING: On free-tier Colab or limited MPS memory, loading all models
    simultaneously may exceed memory. Consider loading one at a time
    with unload_model().

    Args:
        model_keys: List of model keys to load. Defaults to all.
        quantize: Whether to apply 4-bit quantization (CUDA only).
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
    device = get_device()
    lines = [
        "📋 Registered Models:",
        f"   Device: {device.upper()}",
        "=" * 50,
    ]
    for key, config in MODEL_REGISTRY.items():
        lines.append(f"\n  {key}:")
        lines.append(f"    Name: {config.name}")
        lines.append(f"    HF ID: {config.hf_id}")
        lines.append(f"    Description: {config.description}")
        lines.append(f"    Max Context: {config.max_context}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_model_info())
