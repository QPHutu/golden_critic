# Copyright 2026 OpenAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import compute_gdpo_outcome_advantage


def _inputs():
    # Two response groups.  In the second group the two rewards disagree,
    # which distinguishes GDPO from normalizing their sum first.
    return {
        "token_level_rewards": torch.zeros((4, 2)),
        "response_mask": torch.ones((4, 2)),
        "index": np.asarray(["a", "a", "b", "b"], dtype=object),
        "config": {"gdpo_reward_keys": ["hard", "principle"]},
        "non_tensor_batch": {
            "hard": np.asarray([0.0, 1.0, 0.0, 1.0]),
            "principle": np.asarray([0.0, 1.0, 1.0, 0.0]),
        },
        "batch": {
            "prompts": torch.zeros((4, 3), dtype=torch.long),
            "attention_mask": torch.ones((4, 5), dtype=torch.long),
        },
    }


def test_gdpo_decouples_reward_components():
    advantages, returns = compute_gdpo_outcome_advantage(**_inputs())
    assert torch.isfinite(advantages).all()
    assert torch.equal(advantages, returns)
    # Group a has aligned objectives; group b has cancelling objectives.
    assert advantages[0, 0] < 0 < advantages[1, 0]
    assert torch.allclose(advantages[2], torch.zeros(2))
    assert torch.allclose(advantages[3], torch.zeros(2))


def test_gdpo_rejects_mismatched_component_weights():
    inputs = _inputs()
    inputs["config"]["gdpo_reward_weights"] = [1.0]
    with pytest.raises(AssertionError, match="one entry per reward key"):
        compute_gdpo_outcome_advantage(**inputs)
