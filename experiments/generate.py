"""Generates watermarked texts (continuations of C4 articles, or answers to open questions) and saves tokens and settings to a JSON Lines file."""

import argparse
import json
import time
from pathlib import Path

from textgrain.block_ot import removed_fraction, textgrain_chooser
from textgrain.data import continuation_message, load_c4_prompts, load_questions
from textgrain.greenlist import greenlist_chooser
from textgrain.gumbel import gumbel_chooser
from textgrain.keys import load_key, new_key, save_key
from textgrain.model import encode_prompt, get_device, load_model
from textgrain.sampling import TokenChooser, generate_batch, sampler


def make_chooser(args: argparse.Namespace, key: bytes, seed: int, log: list) -> TokenChooser:
    """Takes the command-line settings, the key and a seed and returns the token chooser of the requested method."""
    if args.method == "none":
        return sampler(seed)
    if args.method == "gumbel":
        return gumbel_chooser(key, args.window, seed)
    if args.method == "greenlist":
        return greenlist_chooser(key, args.window, args.gamma, args.delta, seed)
    return textgrain_chooser(key, args.window, args.blocks, args.columns, args.beta, seed, log=log)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--method", choices=["none", "gumbel", "greenlist", "textgrain"], required=True)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--prompts", choices=["c4", "questions"], default="c4")
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1, help="answers generated per prompt, each with its own seed")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-eos", action="store_true", help="forbid end-of-text so that every text has max-new-tokens tokens")
    parser.add_argument("--key-file", default="results/secret.key")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    key_path = Path(args.key_file)
    if not key_path.exists():
        key_path.parent.mkdir(parents=True, exist_ok=True)
        save_key(new_key(), key_path)
    key = load_key(key_path)

    model, tokenizer = load_model(args.model, get_device())
    if args.prompts == "c4":
        prompts = load_c4_prompts(tokenizer, args.n_prompts, completion_tokens=args.max_new_tokens)
        rows = [{**row, "prompt_index": i, "message": continuation_message(row["prefix"])} for i, row in enumerate(prompts)]
    else:
        questions = load_questions(args.n_prompts)
        rows = [{"prefix": q, "human_ids": [], "prompt_index": i, "message": q} for i, q in enumerate(questions)]
    rows = [row for row in rows for _ in range(args.samples)]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in vars(args).items() if k not in ("out", "key_file")}

    start = time.perf_counter()
    with out_path.open("w") as f:
        for first in range(0, len(rows), args.batch_size):
            batch = rows[first : first + args.batch_size]
            logs = [[] for _ in batch]
            choosers = [make_chooser(args, key, args.seed + first + i, logs[i]) for i in range(len(batch))]
            prompts = [encode_prompt(tokenizer, row["message"]) for row in batch]
            records = [[] for _ in batch]
            responses = generate_batch(
                model, prompts, choosers, tokenizer.pad_token_id, args.max_new_tokens, ban_stop=args.no_eos, records=records
            )
            for i, (row, response) in enumerate(zip(batch, responses)):
                record = {
                    "index": first + i,
                    "prompt_index": row["prompt_index"],
                    "settings": settings,
                    "prefix": row["prefix"],
                    "human_ids": row["human_ids"],
                    "ids": response,
                    "text": tokenizer.decode(response, skip_special_tokens=True),
                    "removed_fraction": removed_fraction(logs[i]) if logs[i] else None,
                    "entropies": [round(h, 4) for h, _ in records[i]],
                    "stop_probs": [round(q, 4) for _, q in records[i]],
                }
                f.write(json.dumps(record) + "\n")
            print(f"{first + len(batch)}/{len(rows)} texts, {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
