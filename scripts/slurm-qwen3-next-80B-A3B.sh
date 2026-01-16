#!/bin/bash
#SBATCH --job-name=slime-qwen-train
#SBATCH --nodes=4                    # Total number of nodes
#SBATCH --ntasks-per-node=1          # One main task per node (for Ray startup)
#SBATCH --gpus-per-node=8            # GPUs per node
#SBATCH --cpus-per-task=64           # Adjust based on your node's CPU count
#SBATCH --exclusive                  # Request exclusive access to nodes
#SBATCH --output=logs/%x-%j.out      # Standard output log
#SBATCH --error=logs/%x-%j.err       # Standard error log
#SBATCH --time=48:00:00              # Time limit (adjust as needed)

set -e

# --- 1. Environment & Node Setup ---

export BASE_FOLDER=/mnt/slurm
# Ensure SLIME_DIR is set (if not set in environment, defaults to BASE_FOLDER/slime or similar)
# export SLIME_DIR=${SLIME_DIR:-/mnt/slurm/slime_dir_placeholder} 

if [ -z "${BASE_FOLDER}" ]; then
  echo "BASE_FOLDER is not set."
  exit 1
fi

# Get the list of nodes from Slurm
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# The first node is the Head Node
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Nodes: $nodes"
echo "Head Node: $head_node ($head_node_ip)"

export MASTER_ADDR=$head_node_ip
export RAY_ADDRESS=$head_node_ip:6379
# Add head node IP to no_proxy to avoid proxy issues
export no_proxy="127.0.0.1,localhost,${head_node_ip}"
export PYTHONBUFFERED=16

# --- 2. Cleanup & Check ---

echo "Cleaning up previous processes on all nodes..."
# Using srun to run cleanup on all allocated nodes
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 bash -c "pkill -9 sglang; pkill -9 ray; pkill -9 python; rm -rf /tmp/ray/*" || true
sleep 3

# Check NVLink on the head node (assumed similar across nodes)
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# --- 3. Start Ray Cluster ---

echo "Starting Ray HEAD on $head_node"
# Start Head (runs on the script execution node, which is Node 0)
ray start --head --node-ip-address="${head_node_ip}" --port=6379 \
    --num-gpus=8 --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265 &
sleep 10

echo "Starting Ray WORKERS"
# Loop through remaining nodes and start workers via srun
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i=1; i<=worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "  -> Starting worker on $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "${head_node_ip}:6379" \
        --num-gpus=8 --disable-usage-stats &
done

echo "Waiting for Ray cluster to initialize..."
sleep 20
ray status

# --- 4. Training Arguments ---

# Directory of this script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Source your model config (Make sure this path works inside Slurm)
source "${SCRIPT_DIR}/models/qwen3-next-80B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${BASE_FOLDER}/models/Qwen3-Next-80B-A3B-Thinking
   --ref-load ${SLIME_DIR}/resource/Qwen3-Next-80B-A3B-Thinking_torch_dist
   --load ${SLIME_DIR}/ckpts/Qwen3-Next-80B-A3B-Thinking_slime/
   --save ${SLIME_DIR}/ckpts/Qwen3-Next-80B-A3B-Thinking_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${BASE_FOLDER}/data/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime l/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 2
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator gspo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 4e-4
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-next-80B-A3B-test
   --wandb-key 215aba97807e844aa3fd6d9cf554c28ac64edec8
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.8
   --sglang-ep-size 8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 2
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 3
   --sglang-enable-draft-weights-cpu-backup
   --sglang-max-running-requests 512
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type flex
   --moe-enable-deepep
)

# --- 5. Job Submission ---

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/mnt/slurm/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
  }
}"

echo "Submitting Ray Job..."

# NOTE: We use http://127.0.0.1:8265 because we are currently ON the head node
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 4 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}