#!/usr/bin/env python3
"""Mitra — Tenglish friend chatbot inference with LoRA adapter."""
import argparse
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


def main():
    parser = argparse.ArgumentParser(description="Chat with Mitra")
    parser.add_argument("--base", action="store_true", help="Use base model without adapter (for comparison)")
    parser.add_argument("--model", default="sarvamai/sarvam-2b-v0.5", help="Base model path")
    parser.add_argument("--adapter", default="./adapters", help="Adapter path")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens per response")
    args = parser.parse_args()

    print("Loading model...")
    if args.base:
        print("[BASE MODE] Loading without adapter")
        model, tokenizer = load(args.model)
    else:
        print(f"[MITRA MODE] Loading with adapter from {args.adapter}")
        model, tokenizer = load(args.model, adapter_path=args.adapter)

    sampler = make_sampler(temp=0.7, top_p=0.9)

    print("\n" + "=" * 50)
    print("  Mitra — Your Tenglish Friend 🎬🎵")
    print("  Type 'bye' or Ctrl+C to exit")
    print("=" * 50 + "\n")

    history = []
    MAX_HISTORY = 4  # Keep last 4 turns

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("bye", "exit", "quit"):
                print("Mitra: Bye bro! Take care, malli kaliddham! 👋")
                break

            history.append(user_input)
            if len(history) > MAX_HISTORY:
                history = history[-MAX_HISTORY:]

            # Build prompt with history
            prompt_parts = []
            for i, msg in enumerate(history):
                prompt_parts.append(f"<s>[INST] {msg} [/INST]")

            prompt = "".join(prompt_parts)
            response = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
            )

            # Clean response
            response = response.strip()
            if "</s>" in response:
                response = response.split("</s>")[0].strip()

            print(f"Mitra: {response}\n")

    except KeyboardInterrupt:
        print("\nMitra: Bye bro! Take care! 👋")


if __name__ == "__main__":
    main()
