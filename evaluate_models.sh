#!/usr/bin/env bash
set -euo pipefail

# Fixed scale (per your instructions, do NOT change)
SCALE="1.5B"

# Models from the screenshot
MODELS=(
  "GPT_Score3-GPT4.1-mini:1.5B_alpha:0.1"
  "GPT_Score3-GPT4.1-nano:1.5B_alpha:0.2"
  "GPT_Score3-GPT4o-mini:1.5B_alpha:0.2"
  "sigmoid:1.5B_alpha:0.1"
  "sigmoid:1.5B_alpha:0.2"
  "sigmoid:1.5B_alpha:0.4"
)

# Datasets to evaluate on
DATASETS=(
  "openai/gsm8k"
  "di-zhang-fdu/MATH500"
  "datasets/converted_aime_dataset"
)

for ds in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    cmd="python evaluate_model.py --model_path='runs/${model}/' --dataset=${ds} --scale=${SCALE}"
    echo ">>> ${cmd}"
    eval "${cmd}"
    echo
  done
done
