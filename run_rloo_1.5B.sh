#!/bin/bash

#SBATCH --mem=100g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --job-name=run
#SBATCH --partition=ghx4
#SBATCH --time=24:00:00      # hh:mm:ss for the job
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out

echo "job is starting on `hostname`"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

MODEL_SCALE='1.5B'
REWARD_TYPE='GPT_Score3'
ALPHA=0.1 # This controls the penalty for longer correct respones. Increase to penalize longer responses.
WANDB_KEY="65c076a4c41e029288e44594bce4d21f2d539512" # Provide your wandb key here before running
CHECK_EOS='--check_eos'
SCHEDULER_TYPE='warmup_with_constant_lr' # can be cosine otherwise
OPENAI_KEY=YOUR_KEY_HERE
MODEL_API='gpt-4.1-mini-2025-04-14'

PRETRAIN='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'  
ACTOR_NUM_GPUS=2
REF_NUM_GPUS=2
VLLM_NUM_ENGINES=2
ACTOR_LEARNING_RATE=5e-6
INIT_KL_COEF=0.001
MIN_P=0
MAX_EPOCHS=1
TOKENIZER='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
NUM_EPISODES=1
GENERATE_MAX_LEN=32000
SAVE_STEPS=10
SEED=42

RUN_NAME=$REWARD_TYPE":"$MODEL_SCALE"_alpha:"$ALPHA
INPUT_KEY="problem"
DATASET='datasets/compression_dataset'
BASE_PROJECT_DIR='/mnt/razvandu/eff_grpo/other_techniques/efficient-reasoning/runs' # Change this to the path of the project directory
RM_ADDRESS="0.0.0.0:12366"
SAVE_PATH="$BASE_PROJECT_DIR/$RUN_NAME"
CKPT_PATH="$SAVE_PATH"

echo "Using: ($DATASET) logging run to ($RUN_NAME)"

# stop if any previous instances are running
ray stop -f
# launch the master node of ray in container
ray start --head --node-ip-address 0.0.0.0 --num-gpus $NUM_GPUS --ray-debugger-external

# launch reward server
python -m reward_server.math_server \
  --address $RM_ADDRESS \
  --dataset $DATASET \
  --tokenizer $TOKENIZER \
  --openai_key $OPENAI_KEY \
  --reward_type $REWARD_TYPE \
  --model_api $MODEL_API \
  --alpha $ALPHA \
  $CHECK_EOS \
  &> logs/server_still-3-GPT_Separated-GPT4.1-mini-$SLURM_JOB_ID.all &

python -m openrlhf.cli.train_ppo_ray \
  --advantage_estimator rloo \
  --n_samples_per_prompt 8 \
  --max_epochs $MAX_EPOCHS \
  --remote_rm_url http://$RM_ADDRESS/query \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node $REF_NUM_GPUS \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node $ACTOR_NUM_GPUS \
  --vllm_num_engines $VLLM_NUM_ENGINES \
  --vllm_tensor_parallel_size 1 \
  --max_ckpt_num 10 \
  --num_episodes $NUM_EPISODES \
  --colocate_critic_reward \
  --colocate_actor_ref \
  --pretrain $PRETRAIN \
  --wandb_run_name $RUN_NAME \
  --save_path $SAVE_PATH \
  --ckpt_path $CKPT_PATH \
  --save_steps $SAVE_STEPS \
  --prompt_data_probs 1.0 \
  --scheduler_type $SCHEDULER_TYPE \
  --min_p $MIN_P \
  --micro_train_batch_size 1 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 1 \
  --rollout_batch_size 32 \
  --max_samples 3200 \
  --prompt_max_len 512 \
  --generate_max_len $GENERATE_MAX_LEN \
  --zero_stage 2 \
  --bf16 \
  --seed $SEED \
  --actor_learning_rate $ACTOR_LEARNING_RATE \
  --init_kl_coef $INIT_KL_COEF \
  --prompt_data $DATASET \
  --input_key $INPUT_KEY \
  --input_template $'<｜begin▁of▁sentence｜><｜User｜>Please reason step by step, and put your final answer within \\boxed{{}}. Question: {}<｜Assistant｜>' \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb $WANDB_KEY
