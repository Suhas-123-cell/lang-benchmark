#!/bin/bash
# Mitra — Environment Setup (Step 1)

set -e

echo "=== Creating virtual environment ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install mlx-lm huggingface_hub datasets

echo "=== Logging into HuggingFace ==="
echo "Paste your HF token when prompted:"
huggingface-cli login

echo "=== Downloading Sarvam-2B ==="
python3 -c "
from mlx_lm import load, generate
print('Loading sarvamai/sarvam-2b-v0.5...')
model, tokenizer = load('sarvamai/sarvam-2b-v0.5')
prompt = 'Hyderabad is a beautiful city'
response = generate(model, tokenizer, prompt=prompt, max_tokens=50)
print('Test output:', response)
print('Setup complete!')
"
