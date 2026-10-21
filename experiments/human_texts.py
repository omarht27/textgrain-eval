"""Saves C4 human texts, disjoint from the experiment prompts, to measure false positives."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from textgrain.data import load_c4_prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--skip", type=int, default=200, help="articles already used as prompts")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", default="results/human.jsonl")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = load_c4_prompts(tokenizer, args.skip + args.n)[args.skip :]
    with Path(args.out).open("w") as f:
        for i, row in enumerate(rows):
            record = {"index": i, "prompt_index": i, "settings": {"method": "none", "window": 2}, "prefix": row["prefix"], "human_ids": row["human_ids"], "ids": []}
            f.write(json.dumps(record) + "\n")
    print(f"{len(rows)} human texts saved")


if __name__ == "__main__":
    main()
