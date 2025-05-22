#!/usr/bin/env bash
set -euo pipefail

# ─── fixed values ────────────────────────────────────────────────────
SCALE="1.5B"

MODELS=(

  "l3lab/L1-Qwen-1.5B-Max"
  "l3lab/L1-Qwen-1.5B-Exact"

  #"runs/still-3:sigmoid:1.5B_alpha:0.9/"
  #"runs/still-3:GPT_Separated-GPT4.1-mini:1.5B_alpha:0.1/"
  #"runs/still-3:cosine_default-14336:2.0:1.0:-10.0:0.0:-10.0:1.5B_alpha:0.1/"

  #"runs/still-3:sigmoid:1.5B_alpha:0.4/"
  #"runs/still-3:sigmoid:1.5B_alpha:0.6/"
  
  #"runs/still-3:GPT_Score3-GPT4.1-mini:1.5B_alpha:0.1/"

  #"runs/still-3:sigmoid:1.5B_alpha:0.1/"
  #"runs/still-3:sigmoid:1.5B_alpha:0.2/"

  #"RUC-AIBOX/STILL-3-1.5B-preview"


  #"runs/sigmoid:1.5B_alpha:0.6/"


  #"runs/sigmoid:1.5B_alpha:0.8/"
  
  #"runs/GPT_Score3-GPT4.1-mini-CORB:1.5B_alpha:0.2/"
  #"runs/GPT_Score3-GPT4.1-mini-COMB:1.5B_alpha:0.2/"
  #"runs/sigmoid:1.5B_alpha:0.1/"
  #"runs/sigmoid:1.5B_alpha:0.2/"
  #"runs/sigmoid:1.5B_alpha:0.4/"
  #"runs/cosine:14336:2.0:1.0:-10.0:0.0:-10.0:1.5B_alpha:0.4/"
  
  #"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

  #"Qwen/Qwen2.5-1.5B-Instruct"

  #"runs/GPT_Score3-GPT4.1-nano:1.5B_alpha:0.2/"
  #"runs/GPT_Score3-GPT4o-mini:1.5B_alpha:0.2/"

  #"../../saves/Qwen/Qwen2.5-1.5B-Instruct/GRPO-GPTScore_3PlusAcc/checkpoint-1251/"
  #"../../saves/Qwen/Qwen2.5-1.5B-Instruct/GRPO-training-Vanilla/checkpoint-1251/"
  #"Qwen/Qwen2.5-1.5B-Instruct"
  #"runs/deepscaler:sigmoid:1.5B_alpha:0.2/"
  #"../../saves/Qwen/Qwen2.5-1.5B-Instruct/GRPO-training-Cosine-400/checkpoint-1251/"
  #"../../saves/Qwen/Qwen2.5-1.5B-Instruct/GRPO-training-EffReasoning-0.1/checkpoint-1251/"
  #"../../saves/Qwen/Qwen2.5-1.5B-Instruct/GRPO-training-AccGPT-2/checkpoint-1251/"
  #"../../saves/Qwen/Qwen2.5-1.5B-Instruct/GRPO-training-GPTScore_3/checkpoint-1251/"
  #"../../saves/Qwen/Qwen2.5-1.5B-Instruct/GRPO-training-GPTScore_3-LinearReward/checkpoint-1251/"
  #"runs/deepscaler:sigmoid:1.5B_alpha:0.4/"
  #"agentica-org/DeepScaleR-1.5B-Preview"
  #"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  #"runs/cosine:5000:2.0:1.0:-10.0:0.0:-10.0:1.5B_alpha:0.4/"
  #"runs/cosine:14336:2.0:1.0:-10.0:0.0:-10.0:1.5B_alpha:0.4/"
  #"runs/GPT_Score3-GPT4.1-mini-CORB:1.5B_alpha:0.2/"
  #"runs/sigmoid:1.5B_alpha:0.1/"
  #"runs/sigmoid:1.5B_alpha:0.2/"
  #"runs/sigmoid:1.5B_alpha:0.4/"
  #"runs/GPT_Score3-GPT4o-mini:1.5B_alpha:0.2/"
  #"runs/GPT_Score3-GPT4.1-nano:1.5B_alpha:0.2/"
  #"runs/GPT_Score3-GPT4.1-mini:1.5B_alpha:0.1/"
  #"runs/GPT_Score3-GPT4.1-mini-COMB:1.5B_alpha:0.2/"
)

DATASETS=(
  #"openai/gsm8k"
  #"di-zhang-fdu/MATH500"
  #"TIGER-Lab/MMLU-Pro"
  #"Idavidrein/gpqa"
  "TIGER-Lab/TheoremQA"
)

#DATASETS=(
#  "isaiahbjork/cot-logic-reasoning"
#  "opencompass/AIME2025"
#  "datasets/compression_dataset"
#)

# ─── helper: build HF folder if it doesn't exist ─────────────────────
build_hf_dir () {              # args: <model_root> <step>
  local root="$1" ; local step="$2"
  local orig_dir="${root}/_actor"
  local tag="global_step${step}"
  local zero_dir="${root}/_actor/global_step${step}"
  local hf_dir="${root}/hf_step${step}"
  local out_bin="${hf_dir}/pytorch_model.bin"

  [[ -f "${out_bin}" ]] && return          # already built

  echo "[prep] ${root##*/} step ${step}: merging ZeRO shards → HF folder"
  mkdir -p "${hf_dir}"

  # 1) merge ZeRO shards → pytorch_model.bin
  python "${root}/_actor/zero_to_fp32.py"  "${orig_dir}"  "${out_bin}"  --tag "${tag}"

  # 2) copy meta JSONs so vLLM/HF can load the folder
  for f in config.json generation_config.json special_tokens_map.json \
           tokenizer_config.json tokenizer.json; do
    cp -n "${root}/${f}" "${hf_dir}/"      # -n = don't overwrite if already there
  done
}

# ─── main loop ────────────────────────────────────────────────────────
for ds in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    cmd="python evaluate_model.py --model_path='${model}' --dataset=${ds} --scale=${SCALE}"
    echo ">>> ${cmd}"
    eval "${cmd}"
    echo
    #for step in {100..50..-10}; do          # 50 60 70 80 90 100
    #  model_root="runs/${model}"
    #
    #  build_hf_dir "${model_root}" "${step}"

    #  cmd="python evaluate_model.py \
    #        --model_path='${model_root}/hf_step${step}' \
    #        --dataset='${ds}' \
    #        --scale='${SCALE}'"

    #  echo ">>> ${cmd}"
    #  eval "${cmd}"
    #  echo
    done
  done
done
