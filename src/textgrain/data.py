"""Prompts and human reference texts."""

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

INSTRUCTION = "Continue the following news article with about 300 words. Write only the continuation, in the same style.\n\n"


def load_c4_prompts(tokenizer: PreTrainedTokenizerBase, n: int, prefix_tokens: int = 100, completion_tokens: int = 400) -> list[dict]:
    """Takes a tokenizer and returns n C4 (realnewslike) articles split into a prefix of prefix_tokens and a human continuation of completion_tokens."""
    rows = []
    for row in load_dataset("allenai/c4", "realnewslike", split="validation", streaming=True):
        ids = tokenizer(row["text"], add_special_tokens=False).input_ids
        if len(ids) < prefix_tokens + completion_tokens:
            continue
        prefix = tokenizer.decode(ids[:prefix_tokens])
        rows.append({"prefix": prefix, "human_ids": ids[prefix_tokens : prefix_tokens + completion_tokens]})
        if len(rows) == n:
            break
    return rows


def continuation_message(prefix: str) -> str:
    """Takes the beginning of an article and returns the user message asking the model to continue it."""
    return INSTRUCTION + prefix
