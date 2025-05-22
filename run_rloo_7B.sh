#!/bin/bash

#SBATCH --mem=300g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --job-name=run
#SBATCH --partition=ghx4
#SBATCH --time=24:00:00      # hh:mm:ss for the job
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out

echo "job is starting on `hostname`"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${NUM_GPUS}" --block &
# __doc_head_ray_end__

# __doc_worker_ray_start__
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${NUM_GPUS}" --block &
    sleep 5
done


MODEL_SCALE='7B'
REWARD_TYPE='sigmoid'
ALPHA=0.1
WANDB_KEY=""
CHECK_EOS='--check_eos'
SCHEDULER_TYPE='warmup_with_constant_lr' # can be cosine otherwise


PRETRAIN='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
ACTOR_NUM_GPUS=4
REF_NUM_GPUS=4
VLLM_NUM_ENGINES=4
ACTOR_LEARNING_RATE=2e-6
INIT_KL_COEF=0.001
MIN_P=0
MAX_EPOCHS=1
TOKENIZER='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
NUM_EPISODES=1
GENERATE_MAX_LEN=32000
SAVE_STEPS=10
SEED=42
OPENAI_KEY=YOUR_KEY_HERE
MODEL_API='gpt-4.1-mini-2025-04-14'

RUN_NAME="scale:"$MODEL_SCALE"_alpha:"$ALPHA
INPUT_KEY="problem"
DATASET='datasets/compression_dataset'
BASE_PROJECT_DIR=$PWD
ADDRESS=$head_node_ip:12345
SAVE_PATH="$BASE_PROJECT_DIR/$RUN_NAME"
CKPT_PATH="$SAVE_PATH"

echo "Using: ($DATASET) logging run to ($RUN_NAME)"

# launch gsm_server
python -m reward_server.math_server \
    --address $ADDRESS \
    --dataset $DATASET \
    --tokenizer $TOKENIZER \
    --reward_type $REWARD_TYPE \
    --alpha $ALPHA \
    --openai_key $OPENAI_KEY \
    --model_api $MODEL_API \
    $CHECK_EOS \
    1> logs/server$SLURM_JOB_ID.out 2> logs/server$SLURM_JOB_ID.err&

python -m openrlhf.cli.train_ppo_ray \
    --advantage_estimator rloo \
    --n_samples_per_prompt 8 \
    --max_epochs $MAX_EPOCHS \
    --remote_rm_url http://$ADDRESS/query \
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
    --zero_stage 3 \
    --bf16 \
    --seed $SEED \
    --actor_learning_rate $ACTOR_LEARNING_RATE \
    --init_kl_coef $INIT_KL_COEF \
    --prompt_data $DATASET \
    --input_key $INPUT_KEY \
    --input_template $'<｜begin▁of▁sentence｜><｜User｜>Please reason step by step, and put your final answer within \\boxed{{}}. Question: {}<｜Assistant｜>' \
    --flash_attn \
    --gradient_checkpointing \
    --adam_offload \
    --use_wandb $WANDB_KEY
