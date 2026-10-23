"""Modifies watermarked texts (synonyms, round-trip translation, paraphrase) and saves the modified texts and their token ids."""

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import nltk
import numpy as np
import torch
from nltk.corpus import wordnet
from transformers import AutoModelForCausalLM, AutoTokenizer, MarianMTModel, MarianTokenizer

PARAPHRASE_INSTRUCTION = "Paraphrase the following text. Keep the meaning but change the wording and sentence structure. Output only the paraphrase.\n\n"


def substitute_synonyms(text: str, fraction: float, rng: np.random.Generator) -> str:
    """Takes a text and returns it with about a fraction of its words replaced by a WordNet synonym."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if w.isalpha() and len(w) > 3]
    for i in rng.permutation(candidates)[: int(fraction * len(words))]:
        lemmas = {l.name().replace("_", " ") for s in wordnet.synsets(words[i]) for l in s.lemmas()} - {words[i]}
        if lemmas:
            words[i] = sorted(lemmas)[rng.integers(len(lemmas))]
    return " ".join(words)


def split_sentences(text: str) -> list[str]:
    """Takes a text and returns its sentences."""
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


@torch.no_grad()
def translate(sentences: list[str], name: str, batch_size: int = 32) -> list[str]:
    """Takes sentences and a MarianMT model name and returns their translations."""
    tokenizer = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name).to("cuda").eval()
    out = []
    for first in range(0, len(sentences), batch_size):
        batch = tokenizer(sentences[first : first + batch_size], return_tensors="pt", padding=True, truncation=True).to("cuda")
        out += tokenizer.batch_decode(model.generate(**batch, max_new_tokens=256), skip_special_tokens=True)
    return out


def round_trip(texts: list[str]) -> list[str]:
    """Takes English texts and returns them translated to French and back to English, sentence by sentence."""
    sentences = [split_sentences(t) for t in texts]
    flat = [s for group in sentences for s in group]
    back = translate(translate(flat, "Helsinki-NLP/opus-mt-en-fr"), "Helsinki-NLP/opus-mt-fr-en")
    out, position = [], 0
    for group in sentences:
        out.append(" ".join(back[position : position + len(group)]))
        position += len(group)
    return out


@lru_cache(maxsize=1)
def load_paraphraser(name: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Takes an instruction model name and returns its tokenizer (left padding) and the model on the GPU, loaded once."""
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("cuda").eval()
    return tokenizer, model


@torch.no_grad()
def paraphrase(texts: list[str], name: str, batch_size: int = 4) -> list[str]:
    """Takes texts and an instruction model name and returns the model's paraphrases (greedy decoding)."""
    tokenizer, model = load_paraphraser(name)
    out = []
    for first in range(0, len(texts), batch_size):
        messages = [[{"role": "user", "content": PARAPHRASE_INSTRUCTION + t}] for t in texts[first : first + batch_size]]
        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        generated = model.generate(**batch, max_new_tokens=600, do_sample=False)
        out += tokenizer.batch_decode(generated[:, batch.input_ids.shape[1] :], skip_special_tokens=True)
        print(f"  paraphrased {len(out)}/{len(texts)}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--n-texts", type=int, default=100)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--paraphraser", default="microsoft/Phi-3.5-mini-instruct")
    args = parser.parse_args()
    nltk.download("wordnet", quiet=True)
    qwen = AutoTokenizer.from_pretrained(args.tokenizer)

    for path in map(Path, args.files):
        records = [json.loads(line) for line in path.open()][: args.n_texts]
        texts = [r["text"] for r in records]
        rng = np.random.default_rng(0)
        attacked = {f"synonyms_{f}": [substitute_synonyms(t, f, rng) for t in texts] for f in (0.1, 0.2, 0.3)}
        attacked["translation"] = round_trip(texts)
        torch.cuda.empty_cache()
        attacked["paraphrase"] = paraphrase(texts, args.paraphraser)
        torch.cuda.empty_cache()
        out = [
            {
                "index": r["index"],
                "texts": {name: versions[i] for name, versions in attacked.items()},
                "ids": {name: qwen(versions[i], add_special_tokens=False).input_ids for name, versions in attacked.items()},
            }
            for i, r in enumerate(records)
        ]
        path.with_name(path.stem + ".attacks.json").write_text(json.dumps(out))
        print(f"{path.name}: {len(out)} texts attacked", flush=True)


if __name__ == "__main__":
    main()
