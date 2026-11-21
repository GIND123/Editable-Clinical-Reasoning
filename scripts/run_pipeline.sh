#!/usr/bin/env bash
# Full pipeline for one backbone and one DiReCT fold (edit FOLD / configs to sweep).
#   bash scripts/run_pipeline.sh
# Each stage writes to outputs/ or data/; re-running a stage overwrites its outputs.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/check_access.py
clinblueprint prepare-data       --config configs/data/prepare.yaml
clinblueprint build-silver       --config configs/data/silver.yaml
clinblueprint build-silver       --config configs/data/silver.yaml \
      --set input_files='[data/processed/medcasereasoning.test.records.jsonl]' \
      --set output_file=data/silver/medcasereasoning_test.silver.jsonl      # reference graphs for public benchmark
clinblueprint build-silver       --config configs/data/silver.yaml \
      --set input_files='[data/processed/mimic_note.records.jsonl]' \
      --set output_file=data/silver/mimic_note.silver.jsonl
clinblueprint train-sft          --config configs/train/sft_qwen3_4b.yaml
clinblueprint train-cot-sft      --config configs/train/cot_sft.yaml
clinblueprint draft-graphs       --config configs/eval/draft_graphs.yaml
clinblueprint train-readout      --config configs/train/readout.yaml
clinblueprint train-validator    --config configs/train/validator.yaml
clinblueprint train-faithfulness --config configs/train/faithfulness.yaml
clinblueprint train-dpo          --config configs/train/dpo.yaml
clinblueprint build-benchmark    --config configs/eval/clinedit_faith.yaml
clinblueprint evaluate           --config configs/eval/evaluate.yaml
clinblueprint mech               --config configs/eval/mech.yaml
