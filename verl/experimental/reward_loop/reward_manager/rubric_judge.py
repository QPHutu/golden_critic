# Copyright 2026 OpenAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A frozen, generative LLM judge for per-prompt golden rubrics.

The judge model is served by :class:`RewardModelManager`; this manager only
formats requests and converts its structured verdict into scalar reward
components.  Keeping those components separate lets GDPO normalize hard-rule
and principle signals independently.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable
from typing import Any

import aiohttp

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


_NUMBERED_CRITERION = re.compile(r"(?m)^\s*(?:\d+\s*[.)]|[-*])\s+")


def _category(value: Any) -> str:
    value = str(value or "").lower()
    return "principle" if "principle" in value else "hard_rule"


def normalize_rubrics(rubrics: Any) -> list[dict[str, Any]]:
    """Normalize supported rubric encodings into criterion dictionaries.

    OpenRubrics stores a numbered string, whereas several rubric datasets store
    a list of dictionaries with ``criterion`` and optional ``points`` fields.
    A category is always assigned so GDPO receives a stable set of components.
    """
    if isinstance(rubrics, str):
        pieces = [piece.strip() for piece in _NUMBERED_CRITERION.split(rubrics) if piece.strip()]
        # A non-numbered rubric is still a meaningful single criterion.
        raw_criteria: Iterable[Any] = pieces or [rubrics.strip()]
    elif isinstance(rubrics, Iterable) and not isinstance(rubrics, dict | bytes):
        raw_criteria = rubrics
    else:
        raise ValueError("reward_model.rubrics must be a string or a list of criteria")

    criteria = []
    for index, item in enumerate(raw_criteria, start=1):
        if isinstance(item, dict):
            text = item.get("criterion") or item.get("text") or item.get("rubric")
            category = _category(item.get("category") or item.get("type") or text)
            points = item.get("points", 1.0)
        else:
            text = str(item)
            category = _category(text)
            points = 1.0
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"rubric criterion {index} is empty")
        try:
            weight = abs(float(points))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rubric criterion {index} has invalid points: {points!r}") from exc
        criteria.append({"id": index, "criterion": text.strip(), "category": category, "weight": weight})
    if not criteria:
        raise ValueError("reward_model.rubrics must contain at least one criterion")
    return criteria


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object, accepting common markdown code fences."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text[text.find("{") :] if "{" in text else text
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("judge response does not contain a valid JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("judge response JSON must be an object")
    return value


def parse_judge_response(text: str, criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate verdicts exactly against the criteria sent to the judge."""
    payload = _extract_json(text)
    verdicts = payload.get("criteria")
    if not isinstance(verdicts, list):
        raise ValueError("judge response must contain a 'criteria' list")
    by_id = {}
    for verdict in verdicts:
        if not isinstance(verdict, dict) or not isinstance(verdict.get("id"), int):
            raise ValueError("every judge verdict must be an object with an integer id")
        if not isinstance(verdict.get("met"), bool):
            raise ValueError("every judge verdict must contain boolean 'met'")
        if verdict["id"] in by_id:
            raise ValueError("judge response contains duplicate criterion ids")
        by_id[verdict["id"]] = verdict
    expected_ids = {criterion["id"] for criterion in criteria}
    if set(by_id) != expected_ids:
        raise ValueError("judge verdict ids do not exactly match the supplied rubric criteria")
    return [
        {
            **criterion,
            "met": by_id[criterion["id"]]["met"],
            "rationale": str(by_id[criterion["id"]].get("rationale", "")),
        }
        for criterion in criteria
    ]


def aggregate_verdicts(
    verdicts: list[dict[str, Any]], hard_rule_weight: float, principle_weight: float
) -> dict[str, float]:
    """Return stable reward components for scalar PPO/GRPO and GDPO."""
    components = {}
    for category, name in (("hard_rule", "rubric_hard_reward"), ("principle", "rubric_principle_reward")):
        selected = [verdict for verdict in verdicts if verdict["category"] == category]
        denominator = sum(verdict["weight"] for verdict in selected)
        components[name] = (
            sum(verdict["weight"] * float(verdict["met"]) for verdict in selected) / denominator
            if denominator > 0
            else 0.0
        )

    available = []
    if any(verdict["category"] == "hard_rule" for verdict in verdicts):
        available.append((hard_rule_weight, components["rubric_hard_reward"]))
    if any(verdict["category"] == "principle" for verdict in verdicts):
        available.append((principle_weight, components["rubric_principle_reward"]))
    if not available:
        raise ValueError("no rubric categories were available for aggregation")
    denominator = sum(weight for weight, _ in available)
    if denominator <= 0:
        raise ValueError("at least one rubric category weight must be positive")
    components["score"] = sum(weight * value for weight, value in available) / denominator
    return components


def _messages_to_text(messages: Any) -> str:
    lines = []
    if messages is None:
        return ""
    for message in messages:
        if not isinstance(message, dict):
            lines.append(str(message))
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        lines.append(f"{message.get('role', 'user')}: {content}")
    return "\n".join(lines)


@register("rubric_judge")
class RubricJudgeRewardManager(RewardManagerBase):
    """Score rollouts against golden rubrics with the managed reward model."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        if reward_router_address is None:
            raise ValueError("rubric_judge requires reward.reward_model.enable=true")
        self.reward_router_address = reward_router_address
        self.judge_config = config.reward.reward_model

    def _messages(self, prompt: str, response: str, criteria: list[dict[str, Any]]) -> list[dict[str, str]]:
        criteria_text = "\n".join(
            f"{item['id']}. [{item['category']}] {item['criterion']}" for item in criteria
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a precise rubric evaluator. Judge the response only against the supplied criteria. "
                    "Return JSON only, with this exact schema: "
                    '{"criteria":[{"id":1,"met":true,"rationale":"brief reason"}]}. '
                    "Include one verdict for every criterion and do not add criteria."
                ),
            },
            {
                "role": "user",
                "content": f"<prompt>\n{prompt}\n</prompt>\n<response>\n{response}\n</response>\n"
                f"<rubrics>\n{criteria_text}\n</rubrics>",
            },
        ]

    async def _chat_complete(self, messages: list[dict[str, str]]) -> str:
        request = {
            "model": self.judge_config.model_path,
            "messages": messages,
            "temperature": self.judge_config.judge_temperature,
            "max_tokens": self.judge_config.judge_max_tokens,
        }
        timeout = aiohttp.ClientTimeout(total=self.judge_config.judge_timeout)
        url = f"http://{self.reward_router_address}/v1/chat/completions"
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=request) as response:
                response.raise_for_status()
                payload = await response.json(content_type=None)
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("reward model returned an invalid chat-completions payload") from exc
        if not isinstance(content, str):
            raise ValueError("reward model returned a non-text judge response")
        return content

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        data_item = data[-1:][0]
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][-response_ids.shape[-1] :].sum()
        response = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=True)
        )

        reward_model = data_item.non_tensor_batch.get("reward_model", {}) or {}
        rubrics = reward_model.get("rubrics", reward_model.get("rubric"))
        criteria = normalize_rubrics(rubrics)
        prompt = _messages_to_text(data_item.non_tensor_batch.get("raw_prompt", []))
        messages = self._messages(prompt, response, criteria)

        last_error = None
        raw_judgment = ""
        for attempt in range(self.judge_config.judge_max_retries + 1):
            try:
                raw_judgment = await self._chat_complete(messages)
                verdicts = parse_judge_response(raw_judgment, criteria)
                components = aggregate_verdicts(
                    verdicts,
                    self.judge_config.judge_hard_rule_weight,
                    self.judge_config.judge_principle_weight,
                )
                return {
                    "reward_score": components["score"],
                    "reward_extra_info": {
                        **components,
                        "rubric_verdicts": verdicts,
                        "rubric_judge_raw": raw_judgment,
                        "rubric_judge_status": "ok",
                    },
                }
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                last_error = exc
                if attempt < self.judge_config.judge_max_retries:
                    await asyncio.sleep(2**attempt)

        return {
            "reward_score": 0.0,
            "reward_extra_info": {
                "score": 0.0,
                "rubric_hard_reward": 0.0,
                "rubric_principle_reward": 0.0,
                "rubric_verdicts": [],
                "rubric_judge_raw": raw_judgment,
                "rubric_judge_status": f"fallback: {last_error}",
            },
        }
