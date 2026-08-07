# Copyright 2026 OpenAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import asyncio
from types import SimpleNamespace

import pytest
import torch

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.rubric_judge import (
    RubricJudgeRewardManager,
    aggregate_verdicts,
    normalize_rubrics,
    parse_judge_response,
)


def test_normalize_openrubrics_text_and_aggregate_components():
    criteria = normalize_rubrics(
        "1. The answer is in English. [Hard Rule]\n2. The answer is concise. [Principle]"
    )
    assert [(item["id"], item["category"]) for item in criteria] == [(1, "hard_rule"), (2, "principle")]

    verdicts = parse_judge_response(
        '{"criteria": [{"id": 1, "met": true}, {"id": 2, "met": false}]}', criteria
    )
    assert aggregate_verdicts(verdicts, hard_rule_weight=2.0, principle_weight=1.0) == {
        "rubric_hard_reward": 1.0,
        "rubric_principle_reward": 0.0,
        "score": pytest.approx(2 / 3),
    }


def test_parse_judge_response_requires_exact_criterion_ids():
    criteria = normalize_rubrics("1. Be correct. [Hard Rule]\n2. Be concise. [Principle]")
    with pytest.raises(ValueError, match="do not exactly match"):
        parse_judge_response('{"criteria": [{"id": 1, "met": true}]}', criteria)


def _sample() -> DataProto:
    return DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[11, 12, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        },
        non_tensors={
            "raw_prompt": [[{"role": "user", "content": "Write a haiku."}]],
            "reward_model": [{"rubrics": "1. Use three lines. [Hard Rule]\n2. Be vivid. [Principle]"}],
        },
    )


def test_rubric_judge_manager_returns_gdpo_components_without_http():
    async def run():
        manager = object.__new__(RubricJudgeRewardManager)
        manager.loop = asyncio.get_running_loop()
        manager.tokenizer = SimpleNamespace(decode=lambda token_ids, skip_special_tokens: "a vivid haiku")
        manager.judge_config = SimpleNamespace(
            judge_max_retries=0,
            judge_hard_rule_weight=1.0,
            judge_principle_weight=1.0,
        )

        async def fake_chat_complete(messages):
            assert "Write a haiku." in messages[1]["content"]
            return '{"criteria": [{"id": 1, "met": true}, {"id": 2, "met": false}]}'

        manager._chat_complete = fake_chat_complete
        result = await manager.run_single(_sample())
        assert result["reward_score"] == 0.5
        assert result["reward_extra_info"]["rubric_hard_reward"] == 1.0
        assert result["reward_extra_info"]["rubric_principle_reward"] == 0.0
        assert result["reward_extra_info"]["rubric_judge_status"] == "ok"

    asyncio.run(run())
