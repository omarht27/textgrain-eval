"""Measures the perplexity of generated and human continuations under a larger model."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FIRST_SPECIAL_ID = 151643


@torch.no_grad()
def perplexity(model: AutoModelForCausalLM, context: list[int], tokens: list[int]) -> float:
    """Takes context and continuation token ids and returns exp of the mean negative log-likelihood of the continuation given the context."""
    input_ids = torch.tensor([context + tokens], device=model.device)
    logits = model(input_ids).logits[0, len(context) - 1 : -1].float()
    targets = torch.tensor(tokens, device=model.device)
    return float(torch.nn.functional.cross_entropy(logits, targets).exp())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda").eval()
    for path in map(Path, args.files):
        out = {"generated": [], "human": []}
        for line in path.open():
            record = json.loads(line)
            context = tokenizer(record["prefix"], add_special_tokens=False).input_ids
            generated = [t for t in record["ids"] if t < FIRST_SPECIAL_ID][: args.max_tokens]
            out["generated"].append(perplexity(model, context, generated) if generated else None)
            if path.stem == "none":
                out["human"].append(perplexity(model, context, record["human_ids"][: args.max_tokens]))
        path.with_name(path.stem + ".ppl.json").write_text(json.dumps(out))
        values = sorted(p for p in out["generated"] if p is not None)
        print(f"{path.name}: median perplexity {values[len(values) // 2]:.2f}", flush=True)


if __name__ == "__main__":
    main()
