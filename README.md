## Data Processing

See [dapo.ipynb](dapo.ipynb) and [data.ipynb](data.ipynb).

## Scripts


```bash
# run BPCO on sanity test
bash examples/dppo_trainer/run_sanity_test.sh
# run BPCO+Ans on sanity test
CRITIC_KEY=prompt_critic bash examples/dppo_trainer/run_sanity_test.sh


# run BPCO on DeepScaleR
bash examples/dppo_trainer/run_math_deepscaler.sh
# run BPCO+Ans on DeepScaleR
CRITIC_KEY=prompt_with_answer bash examples/dppo_trainer/run_math_deepscaler.sh
# run BPCO+Sol on DeepScaleR
CRITIC_KEY=prompt_with_solution bash examples/dppo_trainer/run_math_deepscaler.sh
# run BPCO+Ans+Sol on DeepScaleR
CRITIC_KEY=prompt_with_both bash examples/dppo_trainer/run_math_deepscaler.sh
# run group-based baseline on DeepScaleR
ADV_ESTIMATOR=grpo bash examples/dppo_trainer/run_math_deepscaler.sh

# run BPCO on DAPO-Math-17K + Qwen3-30B-A3B-Base
bash examples/dppo_trainer/run_math_megatron.sh
# run BPCO+Ans on DAPO-Math-17K + Qwen3-30B-A3B-Base
CRITIC_KEY=prompt_critic bash examples/dppo_trainer/run_math_megatron.sh
# run group-based baseline on DAPO-Math-17K + Qwen3-30B-A3B-Base
ADV_ESTIMATOR=grpo bash examples/dppo_trainer/run_math_megatron.sh

# run BPCO on OpenRubrics
bash examples/rubric_judge/run_ppo_privileged_critic.sh
# run BPCO+Rubrics on OpenRubrics
CRITIC_KEY=prompt_critic bash examples/rubric_judge/run_ppo_privileged_critic.sh
# run group-based baseline on OpenRubrics
ADV_ESTIMATOR=grpo bash examples/rubric_judge/run_ppo_privileged_critic.sh
```
