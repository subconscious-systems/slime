#!/bin/bash
# ╔═══════════════════════════════════════════════════════════════╗
# ║  🐴✨ HACHIMI TRAINING SCRIPT ✨🐴                            ║
# ║  ～ 哈基米哈基米～ 让我们开始可爱的训练吧喵！～              ║
# ╚═══════════════════════════════════════════════════════════════╝

# (｡◕‿◕｡) 清理旧进程喵～ 让哈基米有干净的跑道！
echo "🧹 哈基米正在打扫跑道喵～ (=^･ω･^=)"
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
echo "✨ 跑道已经干干净净啦喵！"

set -ex

# 🏠 哈基米的家和朋友们的地址喵～
export MASTER_ADDR="10.128.2.102"
export BASE_FOLDER="/mnt/slurm"
export GLOO_SOCKET_IFNAME=bond0

# 🔧 修复 CUDA 路径喵～ 让 GPU 小马们能顺利工作！
# headers 在 targets/x86_64-linux/include 里面哦，不是 include 喵～

# 🏡 检查哈基米的家有没有设置好喵
if [ -z "${BASE_FOLDER}" ]; then
  echo "😿 呜呜～ BASE_FOLDER 没有设置喵！请告诉哈基米 checkpoints 在哪里喵～"
  exit 1
fi

# 📮 检查主人的地址有没有设置好喵
if [ -z "${MASTER_ADDR}" ]; then
  echo "😿 呜呜～ MASTER_ADDR 没有设置喵！请告诉哈基米主节点的地址喵～"
  exit 1
fi

# 📝 防止 ray 缓冲输出喵～ 哈基米想实时看到进度！
export PYTHONBUFFERED=16

# 🔍 检查 NVLink 连接喵～ 看看 GPU 小马们手拉手了没有！
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
    echo "🎉 太棒啦！GPU 小马们都手拉手连接好了喵！"
else
    HAS_NVLINK=0
    echo "🤔 GPU 小马们还没有 NVLink 连接喵～ 但是没关系！"
fi
echo "🔗 HAS_NVLINK: $HAS_NVLINK (发现了 $NVLINK_COUNT 个 NVLink 连接喵～)"

# 📂 找到脚本的位置喵～
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-next-80B-A3B.sh"
echo "📦 加载了模型配置喵～ 哈基米准备好了！"

# ╭─────────────────────────────────────────────────────────────────╮
# │ 🎒 CHECKPOINT 参数喵～ 哈基米的存档点！                         │
# ╰─────────────────────────────────────────────────────────────────╯
HACHIMI_CKPT_ARGS=(
   # 🌟 备用的模型路径喵～ (暂时不用)
   # --hf-checkpoint ${BASE_FOLDER}/models/Qwen3-Next-80B-A3B-Thinking
   # --ref-load ${SLIME_DIR}/resource/Qwen3-Next-80B-A3B-Thinking_torch_dist
   # --load ${SLIME_DIR}/ckpts/Qwen3-Next-80B-A3B-Thinking_slime/
   # --save ${SLIME_DIR}/ckpts/Qwen3-Next-80B-A3B-Thinking_slime/

   # 🐴 哈基米正在用的模型喵！
   --hf-checkpoint ${BASE_FOLDER}/models/tim-next-80B-A3B-SFT
   --ref-load ${SLIME_DIR}/resource/tim-next-80B-A3B-SFT_torch_dist
   --load ${SLIME_DIR}/ckpts/tim-next-80B-A3B_torch_dist_slime-rl/
   --save ${SLIME_DIR}/ckpts/tim-next-80B-A3B_torch_dist_slime-rl/

   # 💾 每 300 步存档一次喵～ 防止哈基米忘记！
   --save-interval 300
)

# ╭─────────────────────────────────────────────────────────────────╮
# │ 🎠 ROLLOUT 参数喵～ 哈基米要跑很多圈！                          │
# ╰─────────────────────────────────────────────────────────────────╯
HACHIMI_ROLLOUT_ARGS=(
   # 📚 训练数据喵～ 哈基米的课本！
   --prompt-data ${BASE_FOLDER}/data/dapo-math-17k/dapo-math-17k-tim-new.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   
   # 🏃 跑步设置喵～ 
   --num-rollout 3000          # 哈基米要跑 3000 圈喵！加油！
   --rollout-batch-size 32     # 每次跑 32 步喵～
   --n-samples-per-prompt 8    # 每个问题回答 8 次喵～
   --rollout-max-response-len 8192   # 最多说 8192 个字喵～
   --rollout-temperature 1     # 创造力温度喵～

   --global-batch-size 256     # 全局批次大小喵～
   --balance-data              # 平衡数据喵～ 公平竞争！
)

# ╭─────────────────────────────────────────────────────────────────╮
# │ 📊 EVAL 参数喵～ 哈基米的考试设置！                             │
# ╰─────────────────────────────────────────────────────────────────╯
HACHIMI_EVAL_ARGS=(
   --eval-interval 20          # 每 20 步考试一次喵～
   --eval-prompt-data aime /mnt/slurm/data/aime-2024/aime-2024.jsonl  # AIME 数学竞赛喵！
   --n-samples-per-eval-prompt 16    # 每道题做 16 遍喵～
   --eval-max-response-len 16384     # 考试可以写更多字喵～
   --eval-top-p 1              # top-p 采样喵～
)

# ╭─────────────────────────────────────────────────────────────────╮
# │ ⚡ PERF 参数喵～ 哈基米的速度配置！                             │
# ╰─────────────────────────────────────────────────────────────────╯
HACHIMI_PERF_ARGS=(
   # 🎪 并行设置喵～ GPU 小马们分工合作！
   --tensor-model-parallel-size 2    # 2 匹小马负责张量喵～
   --sequence-parallel               # 序列也要并行喵～
   --pipeline-model-parallel-size 4  # 4 级流水线喵～
   --context-parallel-size 2         # 上下文并行喵～
   --expert-model-parallel-size 8    # 8 匹专家小马喵～
   --expert-tensor-parallel-size 1   # 专家张量并行喵～

   # 🔄 重计算设置喵～ 节省内存！
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # 📦 动态批次喵～ 灵活调整！
   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192         # 每个 GPU 最多 8192 个 token 喵～
)

# ╭─────────────────────────────────────────────────────────────────╮
# │ 🎯 GRPO 参数喵～ 哈基米的训练策略！                             │
# ╰─────────────────────────────────────────────────────────────────╯
HACHIMI_GRPO_ARGS=(
   --advantage-estimator gspo        # GSPO 优势估计喵～
   #--use-kl-loss                    # KL 损失暂时不用喵～
   --kl-loss-coef 0.00               # KL 损失系数 = 0 喵～
   --kl-loss-type low_var_kl         # 低方差 KL 喵～
   --kl-coef 0.00                    # KL 系数 = 0 喵～
   --entropy-coef 0.00               # 熵系数 = 0 喵～
   --eps-clip 4e-4                   # PPO 裁剪范围喵～ 小心翼翼！
   --ref-update-interval 200         # 每 200 步更新参考模型喵～
)

# ╭─────────────────────────────────────────────────────────────────╮
# │ 📈 OPTIMIZER 参数喵～ 哈基米的学习方法！                        │
# ╰─────────────────────────────────────────────────────────────────╯
HACHIMI_OPTIMIZER_ARGS=(
   --optimizer adam                  # Adam 优化器喵～ 经典！
   --lr 1e-6                         # 学习率超级小喵～ 慢慢学！
   --lr-decay-style constant         # 保持恒定喵～
   --weight-decay 0.1                # 权重衰减喵～
   --adam-beta1 0.9                  # β1 = 0.9 喵～
   --adam-beta2 0.98                 # β2 = 0.98 喵～

   # 💾 CPU 卸载喵～ 省显存！
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer   # 精度感知优化喵～
)

# ╭─────────────────────────────────────────────────────────────────╮
# │ 📊 WANDB 参数喵～ 哈基米的训练日记！                            │
# ╰─────────────────────────────────────────────────────────────────╯
HACHIMI_WANDB_ARGS=(
   --use-wandb                       # 用 WandB 记录喵～
   --wandb-project slime-dev         # 项目名喵～
   --wandb-group qwen3-next-80B-A3B-test  # 分组喵～
   --wandb-key 215aba97807e844aa3fd6d9cf554c28ac64edec8  # 密钥喵～ 保密！
)

# ╭─────────────────────────────────────────────────────────────────╮
# │ 🚀 SGLANG 参数喵～ 哈基米的推理引擎！                           │
# ╰─────────────────────────────────────────────────────────────────╯
HACHIMI_SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8   # 每个引擎 8 块 GPU 喵～
   --sglang-mem-fraction-static 0.8  # 80% 静态显存喵～
   --sglang-ep-size 8                # EP 大小喵～
   
   # 📈 CUDA Graph 批次大小喵～ 加速推理！
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)

   # 🦅 MTP 推测解码喵～ (暂时不用)
   # --sglang-speculative-algorithm EAGLE
   # --sglang-speculative-num-steps 2
   # --sglang-speculative-eagle-topk 1
   # --sglang-speculative-num-draft-tokens 3
   # --sglang-enable-draft-weights-cpu-backup

   --sglang-max-running-requests 512  # 最多 512 个并发请求喵～
)

# ╭─────────────────────────────────────────────────────────────────╮
# │ 🎪 MISC 参数喵～ 哈基米的其他设置！                             │
# ╰─────────────────────────────────────────────────────────────────╯
HACHIMI_MISC_ARGS=(
   # 🎭 Dropout 设置喵～ megatron 默认是 0.1
   --attention-dropout 0.0           # 注意力 dropout = 0 喵～
   --hidden-dropout 0.0              # 隐藏层 dropout = 0 喵～
   
   # ✨ 精度设置喵～ 让模型更准确！
   --accumulate-allreduce-grads-in-fp32   # FP32 梯度累积喵～
   --attention-softmax-in-fp32            # FP32 softmax 喵～
   
   # 💡 注意力后端喵～ MLA 模型要注释掉这行！
   --attention-backend flash              # Flash Attention 喵～ 快快！

   # 🔀 MoE Token 分发喵～
   --moe-token-dispatcher-type alltoall
   # --moe-enable-deepep              # DeepEP 暂时不用喵～
)

# ═══════════════════════════════════════════════════════════════════
# 🌟 启动 Ray 集群喵～ 召唤所有的 GPU 小马！
# ═══════════════════════════════════════════════════════════════════

echo "🚀 哈基米正在启动 Ray 集群喵～ ヾ(≧▽≦*)o"
export no_proxy="127.0.0.1,${MASTER_ADDR}"
export CUDA_HOME=/mnt/slurm/anaconda3/envs/slime
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 👑 启动主节点喵～ 这是哈基米的指挥部！
echo "👑 启动主节点喵～ 地址: ${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# 🐴 召唤其他 GPU 小马们加入喵～
echo "📢 正在召唤其他节点的 GPU 小马们喵～"
for WORKER_IP in 10.128.2.103 10.128.2.104 10.128.2.106; do
#   if [[ "$WORKER_IP" == "$MLP_WORKER_0_HOST" ]]; then
#     continue
#   fi
  echo "🐴 正在唤醒 ${WORKER_IP} 上的 GPU 小马喵～ (=^･ω･^=)"
  ssh -i /home/ubuntu/.ssh/slime_key ubuntu@"${WORKER_IP}" \
    "source /mnt/slurm/anaconda3/bin/activate slime ; export GLOO_SOCKET_IFNAME=bond0 ; export CUDA_HOME=/mnt/slurm/anaconda3/envs/slime ; export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64:\${CUDA_HOME}/lib:/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH ; pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265"
  echo "✅ ${WORKER_IP} 的 GPU 小马已就位喵！"
done
# wait

echo "🎉 所有 GPU 小马都准备好啦喵！让我们开始训练吧！"

# ═══════════════════════════════════════════════════════════════════
# 🌈 构建运行时环境喵～ 设置所有环境变量！
# ═══════════════════════════════════════════════════════════════════

echo "🔧 正在配置运行时环境喵～"
# CUDA_HOME 和 LD_LIBRARY_PATH 是 Triton 找 CUDA 驱动需要的喵～
CONDA_ENV_PATH="/mnt/slurm/anaconda3/envs/slime"
HACHIMI_RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/mnt/slurm/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"CUDA_HOME\": \"${CONDA_ENV_PATH}\",
    \"LD_LIBRARY_PATH\": \"${CONDA_ENV_PATH}/lib64:${CONDA_ENV_PATH}/lib:/usr/lib/x86_64-linux-gnu\",
    \"TRITON_LIBCUDA_PATH\": \"/usr/lib/x86_64-linux-gnu/libcuda.so\"
  }
}"

# ═══════════════════════════════════════════════════════════════════
# 🏇 提交训练任务喵～ 哈基米，冲啊！！！
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  🐴✨ 哈基米哈基米！训练开始啦！✨🐴                        ║"
echo "║  ～ 冲鸭！向着胜利的终点前进喵！～                          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${HACHIMI_RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 4 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${HACHIMI_CKPT_ARGS[@]} \
   ${HACHIMI_ROLLOUT_ARGS[@]} \
   ${HACHIMI_OPTIMIZER_ARGS[@]} \
   ${HACHIMI_GRPO_ARGS[@]} \
   ${HACHIMI_WANDB_ARGS[@]} \
   ${HACHIMI_PERF_ARGS[@]} \
   ${HACHIMI_EVAL_ARGS[@]} \
   ${HACHIMI_SGLANG_ARGS[@]} \
   ${HACHIMI_MISC_ARGS[@]}

echo ""
echo "🎉✨ 哈基米完成任务啦喵！辛苦了！(=^･ω･^=) ✨🎉"
echo ""

# ═══════════════════════════════════════════════════════════════════
# 📝 备用命令喵～ 以防万一需要手动启动 worker
# ═══════════════════════════════════════════════════════════════════
# ssh -i /home/ubuntu/.ssh/slime_key ubuntu@10.128.2.103 \
#     "source /mnt/slurm/anaconda3/bin/activate slime ; pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=10.128.2.102:6379 --num-gpus 8 --node-ip-address 10.128.2.103 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265"