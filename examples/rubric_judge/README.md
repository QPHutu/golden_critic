# Rubric judge rewards

This recipe serves a frozen local instruction model with `reward.reward_model` and uses it as a golden-rubric judge. It does not require an external API.

Prepare the dataset:

```bash
python examples/data_preprocess/openrubrics.py --output_dir /path/to/openrubrics
```

The dataset keeps `prompt` policy-visible and stores the golden rubric in both `reward_model.rubrics` (for the judge) and `prompt_critic` (for a privileged critic). Set `data.prompt_critic_key=prompt_critic` only for PPO/GAE runs with a critic.

For a GDPO run, use explicit rubric components rather than a holistic score:

```bash
bash examples/rubric_judge/run_gdpo.sh \
  DATA_DIR=/path/to/openrubrics \
  POLICY_MODEL=/path/to/policy \
  JUDGE_MODEL=/path/to/frozen-judge
```

`rubric_hard_reward` and `rubric_principle_reward` are independently group-normalized by GDPO. The aggregate `rubric_score` is logged but is intentionally not included in `algorithm.gdpo_reward_keys`, avoiding double counting.

The judge output must be valid JSON and contain one boolean verdict per golden criterion. Failed or malformed responses are retried, then scored as zero and tagged in rollout logs.

For PPO/GAE with privileged critic context, use `run_ppo_privileged_critic.sh`. Its policy rollout sees only `prompt`; its critic receives `prompt_critic` and therefore the golden rubric.
