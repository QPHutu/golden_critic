#!/usr/bin/env bash
# PPO/GAE with a frozen local rubric judge and a rubric-privileged critic.
set -xeuo pipefail

DATA_DIR=${DATA_DIR:?Set DATA_DIR to converted OpenRubrics parquet files}
POLICY_MODEL=${POLICY_MODEL:?Set POLICY_MODEL}
JUDGE_MODEL=${JUDGE_MODEL:?Set JUDGE_MODEL}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
JUDGE_GPUS_PER_NODE=${JUDGE_GPUS_PER_NODE:-2}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gae \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.prompt_critic_key=prompt_critic \
  reward.reward_manager.name=rubric_judge \
  reward.reward_model.enable=True \
  reward.reward_model.mode=rubric_judge \
  reward.reward_model.enable_resource_pool=True \
  reward.reward_model.model_path="$JUDGE_MODEL" \
  reward.reward_model.n_gpus_per_node="$JUDGE_GPUS_PER_NODE" \
  reward.reward_model.nnodes="$NNODES" \
  reward.reward_model.rollout.name=vllm \
  reward.reward_model.rollout.tensor_model_parallel_size="$JUDGE_GPUS_PER_NODE" \
  reward.reward_model.rollout.temperature=0.0 \
  reward.reward_model.rollout.do_sample=False \
  actor_rollout_ref.model.path="$POLICY_MODEL" \
  critic.model.path="$POLICY_MODEL" \
  trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
  trainer.nnodes="$NNODES" \
  "$@"
