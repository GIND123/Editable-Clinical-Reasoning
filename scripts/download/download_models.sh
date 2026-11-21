#!/usr/bin/env bash
# Download model weights to the local HF cache. Accept each model's licence on its HF page first
# (MedGemma / Gemma require accepting Google's terms). Pin revisions in configs/model/*.yaml.
set -euo pipefail
MODELS=(
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
  "google/medgemma-4b-it"      # VERIFY MedGemma 1.5 4B id
  "Qwen/Qwen3-32B"             # teacher (loaded in 4-bit)
)
for m in "${MODELS[@]}"; do
  echo "== $m"
  huggingface-cli download "$m" --exclude "*.gguf" >/dev/null && echo "   ok"
done
# Interpretability artefacts (optional)
# huggingface-cli download google/gemma-scope-2-4b-it
# huggingface-cli download saillab/medgemma-circuit-tools
