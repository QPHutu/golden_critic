# Copyright 2026 OpenAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Convert OpenRubrics to verl RL parquet with critic-only golden rubrics."""

import argparse
import os

import datasets


def make_row(example, index):
    instruction = example["instruction"]
    rubric = example["rubric"]
    prompt = [{"role": "user", "content": instruction}]
    # This field is deliberately separate from ``prompt``.  Setting
    # data.prompt_critic_key=prompt_critic exposes it to the critic only.
    prompt_critic = [
        {
            "role": "system",
            "content": "You are a value critic. Use the following golden rubric as privileged training context. "
            "Do not treat it as part of the user's request.\n<golden_rubric>\n"
            f"{rubric}\n</golden_rubric>",
        },
        *prompt,
    ]
    return {
        "data_source": "OpenRubrics/OpenRubrics",
        "prompt": prompt,
        "prompt_critic": prompt_critic,
        "reward_model": {"style": "rubric", "rubrics": rubric},
        "extra_info": {"index": index, "source": example.get("source", "")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--validation_size", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = datasets.load_dataset("OpenRubrics/OpenRubrics", split="train")
    split = dataset.train_test_split(test_size=args.validation_size, seed=args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    for name, partition in (("train", split["train"]), ("test", split["test"])):
        converted = partition.map(make_row, with_indices=True, remove_columns=partition.column_names)
        converted.to_parquet(os.path.join(args.output_dir, f"{name}.parquet"))


if __name__ == "__main__":
    main()
