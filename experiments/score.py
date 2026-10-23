"""Computes the detection scores of every generated and human text in the results files."""

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from textgrain.detect import greenlist_hits, gumbel_scores, scored_indices, textgrain_scores
from textgrain.keys import load_key

VOCAB_SIZE = 151936


def detector_names(settings: dict) -> list[str]:
    """Takes the settings of a results file and returns the detectors to run on its watermarked texts."""
    if settings["method"] == "gumbel":
        return ["gumbel"]
    if settings["method"] == "greenlist":
        return ["greenlist"]
    if settings["method"] == "textgrain":
        return [f"textgrain_B{settings['blocks']}"]
    return []


def score(tokens: list[int], detector: str, key: bytes, window: int, min_context: int = 0) -> list[float]:
    """Takes token ids and a detector name and returns the score of each scored position."""
    if detector == "gumbel":
        return gumbel_scores(tokens, key, window, VOCAB_SIZE, min_context).tolist()
    if detector == "greenlist":
        return greenlist_hits(tokens, key, window, VOCAB_SIZE, 0.25, min_context).tolist()
    n_blocks = int(detector.split("_B")[1])
    return textgrain_scores(tokens, key, window, VOCAB_SIZE, n_blocks, n_blocks, min_context).tolist()


def score_record(record: dict, key: bytes, negatives: list[str]) -> dict:
    """Takes one text of a results file and returns its scores, under its own detector and, for unwatermarked sources, every detector."""
    window = record["settings"]["window"]
    min_context = record["settings"].get("min_context", 0)
    out = {"index": record["index"], "scores": {}}
    sources = {"generated": record["ids"], "human": record["human_ids"]}
    for source, tokens in sources.items():
        detectors = detector_names(record["settings"]) if source == "generated" else negatives
        if record["settings"]["method"] == "none" and source == "generated":
            detectors = negatives
        for detector in detectors:
            out["scores"][f"{source}/{detector}"] = {
                "t": scored_indices(tokens, window, min_context),
                "y": score(tokens, detector, key, window, min_context),
            }
    return out


def score_attack_file(path: Path, key: bytes) -> None:
    """Takes an attacks file (modified versions of watermarked texts) and saves the score of every modified version under the original detector."""
    source = path.with_name(path.name.replace(".attacks.json", ".jsonl"))
    settings = json.loads(source.open().readline())["settings"]
    (detector,) = detector_names(settings)
    out = []
    for record in json.loads(path.read_text()):
        scores = {}
        for attack, tokens in record["ids"].items():
            scores[attack] = {"t": scored_indices(tokens, settings["window"]), "y": score(tokens, detector, key, settings["window"])}
        out.append({"index": record["index"], "detector": detector, "scores": scores})
    path.with_name(path.name.replace(".json", ".scores.json")).write_text(json.dumps(out))
    print(f"{path.name}: {len(out)} attacked texts scored", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--key-file", default="results/secret.key")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=40, help="texts scored between two saves, so that an interrupted run can resume")
    args = parser.parse_args()
    key = load_key(args.key_file)
    negatives = ["gumbel", "greenlist", "textgrain_B2", "textgrain_B4", "textgrain_B8"]

    for path in map(Path, args.files):
        out_path = path.with_name(path.name.replace(".attacks.json", ".attacks.scores.json") if path.name.endswith(".attacks.json") else path.stem + ".scores.json")
        if out_path.exists():
            continue
        if path.name.endswith(".attacks.json"):
            score_attack_file(path, key)
            continue
        records = [json.loads(line) for line in path.open()]
        human = negatives if records[0]["settings"]["method"] == "none" else []
        partial_path = out_path.with_suffix(".partial")
        scores = json.loads(partial_path.read_text()) if partial_path.exists() else []
        with Pool(args.workers) as pool:
            for first in range(len(scores), len(records), args.chunk):
                scores += pool.map(partial(score_record, key=key, negatives=human), records[first : first + args.chunk])
                partial_path.write_text(json.dumps(scores))
                print(f"{path.name}: {len(scores)}/{len(records)} texts scored", flush=True)
        out_path.write_text(json.dumps(scores))
        partial_path.unlink()


if __name__ == "__main__":
    main()
