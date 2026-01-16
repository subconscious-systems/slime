#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export MASTER_ADDR="10.128.2.102"
export BASE_FOLDER="/mnt/slurm"
export GLOO_SOCKET_IFNAME=bond0

# Set all environment variables directly
export PYTHONPATH="/mnt/slurm/Megatron-LM/"
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export NCCL_NVLS_ENABLE="1"
export no_proxy="127.0.0.1,${MASTER_ADDR}"
export CUDA_HOME="/mnt/slurm/anaconda3/envs/slime"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:/usr/lib/x86_64-linux-gnu"
export TRITON_LIBCUDA_PATH="/usr/lib/x86_64-linux-gnu/libcuda.so"

# if base folder not set raise error
if [ -z "${BASE_FOLDER}" ]; then
  echo "BASE_FOLDER is not set. Please set it to the base directory of your checkpoints."
  exit 1
fi

if [ -z "${MASTER_ADDR}" ]; then
  echo "MASTER_ADDR is not set. Please set it to the master node address."
  exit 1
fi

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
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
   --eval-prompt-data aime /mnt/slurm/data/aime-2024/aime-2024.jsonl
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

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator gspo
   #--use-kl-loss
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

   --sglang-max-running-requests 512
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash

   --moe-token-dispatcher-type flex
   # --moe-enable-deepep
)

# launch the master node of ray in container
ulimit -n 65536 && ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# for WORKER_IP in $(awk '{print $1}' /root/mpi_rack_hostfile); do
for WORKER_IP in 10.128.2.103 10.128.2.104 10.128.2.106; do
  echo "Starting Ray worker on ${WORKER_IP}"
  ssh -i /home/ubuntu/.ssh/slime_key ubuntu@"${WORKER_IP}" \
    "source /mnt/slurm/anaconda3/bin/activate slime && \
     export GLOO_SOCKET_IFNAME=bond0 && \
     export PYTHONPATH=/mnt/slurm/Megatron-LM/ && \
     export CUDA_HOME=/mnt/slurm/anaconda3/envs/slime && \
     export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64:\${CUDA_HOME}/lib:/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH && \
     pkill -9 sglang && ray stop --force && pkill -9 python && \
     ulimit -n 65536 && ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265"
done

# Run train.py directly (no ray job submit) with runtime_env for PYTHONPATH
python3 -c "
import ray
import os

ray.init(address='auto', runtime_env={
    'env_vars': {
        'PYTHONPATH': '/mnt/slurm/Megatron-LM/',
        'CUDA_DEVICE_MAX_CONNECTIONS': '1',
        'NCCL_NVLS_ENABLE': '1',
        'CUDA_HOME': '/mnt/slurm/anaconda3/envs/slime',
        'LD_LIBRARY_PATH': '/mnt/slurm/anaconda3/envs/slime/lib64:/mnt/slurm/anaconda3/envs/slime/lib:/usr/lib/x86_64-linux-gnu',
        'TRITON_LIBCUDA_PATH': '/usr/lib/x86_64-linux-gnu/libcuda.so',
        'GLOO_SOCKET_IFNAME': 'bond0',
    }
})
" && python3 train.py \
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
