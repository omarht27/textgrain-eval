"""Prompts and human reference texts."""

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

INSTRUCTION = "Continue the following news article with about 300 words. Write only the continuation, in the same style, with no introduction.\n\n"


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


SHOWCASE_QUESTIONS = [
    "What is a text watermark?",
    "Explain to a ten-year-old why the sky is blue.",
    "Write a haiku about the sea.",
    "Give three tips to learn a new language.",
]
DOLLY_CATEGORIES = {"open_qa", "general_qa", "brainstorming", "creative_writing"}


def load_questions(n: int) -> list[str]:
    """Returns n open questions: the showcase questions, then Dolly questions that need no context."""
    questions = list(SHOWCASE_QUESTIONS)
    for row in load_dataset("databricks/databricks-dolly-15k", split="train"):
        if len(questions) >= n:
            break
        if row["category"] in DOLLY_CATEGORIES and not row["context"].strip() and len(row["instruction"]) < 200:
            questions.append(row["instruction"].strip())
    return questions[:n]
