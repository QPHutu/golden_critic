#!/usr/bin/env bash
# main_generation_server | text | vLLM rollout | NVIDIA GPUs
# Rollout-only generation with DeepSeek-LLM-7B-Chat.
#
# Single-node (default):
#   bash run_deepseek_llm_7b.sh
#
# Multi-node (e.g. 2 nodes, rollout_tp=16):
#   NNODES=2 ROLLOUT_TP=16 bash run_deepseek_llm_7b.sh

set -xeuo pipefail

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-"/home/aiops/qiph/verl/models/Qwen3-235B-A22B"}
NNODES=${NNODES:-8}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
DTYPE=${DTYPE:-"float16"}

DATA_PATH=${DATA_PATH:-"/home/aiops/qiph/verl/data/deepscaler/aime_2025.parquet"}
OUTPUT_PATH=${OUTPUT_PATH:-"/home/aiops/qiph/verl/data/deepscaler/qwen3-235B_aime_2025.parquet"}

PROMPT_LENGTH=${PROMPT_LENGTH:-2048}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-8192}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((PROMPT_LENGTH + RESPONSE_LENGTH))}
ROLLOUT_TP=${ROLLOUT_TP:-8}
ROLLOUT_DP=${ROLLOUT_DP:-8}
ROLLOUT_EP=${ROLLOUT_EP:-64}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
N_SAMPLES=${N_SAMPLES:-32}
# ---- end user-adjustable ----

python3 -m verl.trainer.main_generation_server \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    data.train_files="${DATA_PATH}" \
    data.prompt_key=prompt \
    +data.output_path="${OUTPUT_PATH}" \
    actor_rollout_ref.nccl_timeout=10800 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=${DTYPE} \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.prompt_length="${PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DP}" \
    actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n="${N_SAMPLES}" "$@"
