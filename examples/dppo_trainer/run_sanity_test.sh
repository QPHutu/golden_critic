#!/usr/bin/env bash
# PPO | text | FSDP training | GPU/NPU
# Canonical PPO (actor + critic) baseline on GSM8K + MATH.

set -xeuo pipefail

########################### user-adjustable ###########################
# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}

# Paths
DATA_ROOT=${DATA_ROOT:-"/home/aiops/qiph/verl"}
MODEL_PATH=${MODEL_PATH:-"${DATA_ROOT}/models/DeepSeek-R1-Distill-Qwen-1.5B"}

CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-$MODEL_PATH}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8096}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

ACTOR_LR=${ACTOR_LR:-1e-6}
CRITIC_LR=${CRITIC_LR:-1e-5}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
ROLLOUT_N=${ROLLOUT_N:-1}
ROLLOUT_VAL_N=${ROLLOUT_VAL_N:-32}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1500}
SAVE_FREQ=${SAVE_FREQ:-20000}
TEST_FREQ=${TEST_FREQ:-50}

CRITIC_KEY=${CRITIC_KEY:-""}
ALGO=${ALGO:-"dppo_tv"}
AGG_MODE=${AGG_MODE:-"seq-mean-token-sum"}
LAM=${LAM:-1.0}

PROJECT_NAME=${PROJECT_NAME:-verl_ppo_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-ret_bdv_${ALGO}_${AGG_MODE}_${CRITIC_KEY}_Lam${LAM}_$(date +%Y%m%d_%H%M)}

# GSM8K_TRAIN_FILE=${GSM8K_TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
# GSM8K_TEST_FILE=${GSM8K_TEST_FILE:-$HOME/data/gsm8k/test.parquet}
# MATH_TRAIN_FILE=${MATH_TRAIN_FILE:-$HOME/data/math/train.parquet}
# MATH_TEST_FILE=${MATH_TEST_FILE:-$HOME/data/math/test.parquet}

CKPTS_DIR=${CKPTS_DIR:-"${DATA_ROOT}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
MATH_TRAIN_FILE=${TRAIN_FILE:-"${DATA_ROOT}/data/sanity_test/math_1460.parquet"}
MATH_TEST_FILE=${TEST_FILE:-"[${DATA_ROOT}/data/sanity_test/aime_2024.parquet,${DATA_ROOT}/data/sanity_test/aime_2025.parquet]"}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
n_devices_per_node=${NDEVICES_PER_NODE:-8}

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=gae
    algorithm.rollout_correction.bypass_mode=True
    algorithm.lam=${LAM}
    +data.prompt_critic_key=${CRITIC_KEY}
    data.train_files=${MATH_TRAIN_FILE}
    data.val_files=${MATH_TEST_FILE}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.loss_agg_mode=${AGG_MODE}
    actor_rollout_ref.actor.policy_loss.loss_mode=${ALGO}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.n=${ROLLOUT_VAL_N}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

CRITIC=(
    critic.model.path="$CRITIC_MODEL_PATH"
    critic.model.use_remove_padding=True
    critic.model.enable_gradient_checkpointing=True
    critic.optim.lr=${CRITIC_LR}
    critic.use_dynamic_bsz=True
    critic.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    critic.fsdp.param_offload=False
    critic.fsdp.optimizer_offload=False
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.default_local_dir=${CKPTS_DIR}
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${CRITIC[@]}" \
    "${TRAINER[@]}" \
    "$@"
