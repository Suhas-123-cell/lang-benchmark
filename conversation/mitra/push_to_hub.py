#!/usr/bin/env python3
"""Push Mitra LoRA adapter to HuggingFace Hub."""
from huggingface_hub import HfApi, create_repo
import os, shutil

# === FILL THESE IN ===
HF_USERNAME = "[YOUR_HF_USERNAME]"  # e.g., "suhas20sh"
REPO_NAME = "mitra-tenglish-sarvam2b"
# ======================

REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
ADAPTER_PATH = "./adapters"

def main():
    api = HfApi()

    print(f"Creating repo: {REPO_ID}")
    create_repo(REPO_ID, repo_type="model", exist_ok=True)

    # Upload adapter files
    print("Uploading adapter weights...")
    api.upload_folder(
        folder_path=ADAPTER_PATH,
        repo_id=REPO_ID,
        path_in_repo=".",
    )

    # Upload model card
    if os.path.exists("MODEL_CARD.md"):
        api.upload_file(
            path_or_fileobj="MODEL_CARD.md",
            path_in_repo="README.md",
            repo_id=REPO_ID,
        )

    # Upload eval results
    if os.path.exists("results/eval_results.txt"):
        api.upload_file(
            path_or_fileobj="results/eval_results.txt",
            path_in_repo="eval_results.txt",
            repo_id=REPO_ID,
        )

    # Upload training data for reproducibility
    if os.path.exists("data/train.jsonl"):
        api.upload_file(
            path_or_fileobj="data/train.jsonl",
            path_in_repo="data/train.jsonl",
            repo_id=REPO_ID,
        )

    print(f"\nDone! View at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
