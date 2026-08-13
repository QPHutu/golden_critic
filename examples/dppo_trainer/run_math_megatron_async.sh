#!/usr/bin/env bash
# PPO | Qwen3-30B-A3B (MoE) | Megatron training | NVIDIA GPUs
# Canonical PPO (actor + critic) baseline on GSM8K + MATH.

set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1

########################### user-adjustable ###########################
INFER_BACKEND=${INFER_BACKEND:-vllm}

# Paths
DATA_ROOT=${DATA_ROOT:-"/home/aiops/qiph/verl"}
MODEL_PATH=${MODEL_PATH:-/home/aiops/qiph/verl/models/Qwen3-30B-A3B-Base}
MCORE_MODEL_PATH=${MCORE_MODEL_PATH:-}

CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-$MODEL_PATH}
NNODES=${NNODES:-2}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}

ACTOR_LR=${ACTOR_LR:-1e-6}
CRITIC_LR=${CRITIC_LR:-5e-6}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

# Megatron parallelism
ACTOR_TP=${ACTOR_TP:-8}
ACTOR_PP=${ACTOR_PP:-1}
ACTOR_VPP=${ACTOR_VPP:-1}
ACTOR_EP=${ACTOR_EP:-8}
ACTOR_CP=${ACTOR_CP:-1}
REF_TP=${REF_TP:-${ACTOR_TP}}
REF_PP=${REF_PP:-${ACTOR_PP}}
REF_VPP=${REF_VPP:-${ACTOR_VPP}}
REF_EP=${REF_EP:-${ACTOR_EP}}
REF_CP=${REF_CP:-${ACTOR_CP}}
CRITIC_TP=${CRITIC_TP:-${ACTOR_TP}}
CRITIC_PP=${CRITIC_PP:-${ACTOR_PP}}
CRITIC_VPP=${CRITIC_VPP:-${ACTOR_VPP}}
CRITIC_EP=${CRITIC_EP:-${ACTOR_EP}}
CRITIC_CP=${CRITIC_CP:-${ACTOR_CP}}
ALL_OFFLOAD=${ALL_OFFLOAD:-True}

# MoE generation (rollout) parallelism — only used when the rollout backend shards experts separately from attention/TP
GEN_MOE_TP=${GEN_MOE_TP:-1}
GEN_MOE_EP=${GEN_MOE_EP:-4}

ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.75}
ROLLOUT_N=${ROLLOUT_N:-1}
ROLLOUT_VAL_N=${ROLLOUT_VAL_N:-32}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1500}
SAVE_FREQ=${SAVE_FREQ:-2500}
TEST_FREQ=${TEST_FREQ:-25}

CRITIC_KEY=${CRITIC_KEY:-""}
ALGO=${ALGO:-"dppo_tv"}
AGG_MODE=${AGG_MODE:-"seq-mean-token-sum"}
LAM=${LAM:-0.4}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-"lagae"}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4000}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16000}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-20000}

if [[ $ADV_ESTIMATOR == "grpo" ]]; then
    TRAIN_BATCH_SIZE=64
    PPO_MINI_BATCH_SIZE=16
    ROLLOUT_N=8
fi

PROJECT_NAME=${PROJECT_NAME:-verl_ppo_moe}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${CRITIC_KEY}_${ADV_ESTIMATOR}_Lam${LAM}_${ALGO}_${AGG_MODE}_$(date +%Y%m%d_%H%M)}

# GSM8K_TRAIN_FILE=${GSM8K_TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
# GSM8K_TEST_FILE=${GSM8K_TEST_FILE:-$HOME/data/gsm8k/test.parquet}
# MATH_TRAIN_FILE=${MATH_TRAIN_FILE:-$HOME/data/math/train.parquet}
# MATH_TEST_FILE=${MATH_TEST_FILE:-$HOME/data/math/test.parquet}

CKPTS_DIR=${CKPTS_DIR:-"${DATA_ROOT}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
ROOT="${DATA_ROOT}/data/dapo/"
MATH_TRAIN_FILE=${TRAIN_FILE:-"${ROOT}/dapo-math-17k-unique.parquet"}
MATH_TEST_FILE=${TEST_FILE:-"[${ROOT}/aime-2025.parquet]"}
# MATH_TEST_FILE=${TEST_FILE:-"[${ROOT}/aime_2025.parquet,${ROOT}/aime_2026.parquet,${ROOT}/brumo_2025.parquet,${ROOT}/hmmt_feb_2026.parquet,${ROOT}/hmmt_nov_2025.parquet]"}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
n_devices_per_node=${NDEVICES_PER_NODE:-8}

[ "${ACTOR_PP}" -gt 1 ] && actor_vpp_override=${ACTOR_VPP} || actor_vpp_override=null
[ "${REF_PP}" -gt 1 ] && ref_vpp_override=${REF_VPP} || ref_vpp_override=null
[ "${CRITIC_PP}" -gt 1 ] && critic_vpp_override=${CRITIC_VPP} || critic_vpp_override=null

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=${ADV_ESTIMATOR}
    algorithm.rollout_correction.bypass_mode=True
    algorithm.lam=${LAM}
    algorithm.norm_adv_by_std_in_grpo=False
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
    actor_rollout_ref.actor.calculate_entropy=True
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${actor_vpp_override}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
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
    +actor_rollout_ref.rollout.moe_tensor_parallel_size=${GEN_MOE_TP}
    actor_rollout_ref.rollout.expert_parallel_size=${GEN_MOE_EP}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP}
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${ref_vpp_override}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${REF_EP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.ref.megatron.use_mbridge=True
)

CRITIC=(
    critic.model.path="$CRITIC_MODEL_PATH"
    critic.model.use_remove_padding=True
    critic.model.enable_gradient_checkpointing=True
    critic.optim.lr=${CRITIC_LR}
    critic.use_dynamic_bsz=True
    critic.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    critic.megatron.tensor_model_parallel_size=${CRITIC_TP}
    critic.megatron.pipeline_model_parallel_size=${CRITIC_PP}
    critic.megatron.virtual_pipeline_model_parallel_size=${critic_vpp_override}
    critic.megatron.expert_model_parallel_size=${CRITIC_EP}
    critic.megatron.context_parallel_size=${CRITIC_CP}
    critic.megatron.param_offload=${ALL_OFFLOAD}
    critic.megatron.optimizer_offload=${ALL_OFFLOAD}
    critic.megatron.grad_offload=${ALL_OFFLOAD}
    critic.megatron.use_mbridge=True
    +critic.megatron.override_transformer_config.recompute_method=uniform
    +critic.megatron.override_transformer_config.recompute_granularity=full
    +critic.megatron.override_transformer_config.recompute_num_layers=1
    +critic.megatron.override_transformer_config.apply_rope_fusion=True
    +critic.megatron.override_transformer_config.masked_softmax_fusion=True
    +critic.megatron.override_transformer_config.bias_activation_fusion=True
    +critic.megatron.override_transformer_config.bias_dropout_fusion=True
    +critic.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +critic.megatron.override_transformer_config.deallocate_pipeline_outputs=True
    +critic.megatron.override_transformer_config.persist_layer_norm=True
    +critic.megatron.override_transformer_config.moe_grouped_gemm=True
    +critic.megatron.override_transformer_config.moe_permute_fusion=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=15
    trainer.log_val_generations=1
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

EXTRA=(
    model_engine=megatron
)

if [ -n "$MCORE_MODEL_PATH" ]; then
    EXTRA+=(
        actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH}
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=True
        actor_rollout_ref.ref.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH}
        actor_rollout_ref.ref.megatron.use_dist_checkpointing=True
        critic.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH}
        critic.megatron.use_dist_checkpointing=True
    )
fi

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${CRITIC[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
