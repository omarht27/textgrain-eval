"""Turns the results and scores files into tables, figures and text examples."""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, gamma, kstest

from textgrain.analysis import detection_rate, natural_length

RESULTS = Path("results")
FIGURES = Path("figures")
LENGTHS = [25, 50, 100, 200, 400]
ALPHA = 0.01
CONFIGS = {
    "gumbel": "Gumbel-max",
    "greenlist_d2": "green list (delta=2)",
    "textgrain_B2_b0.25": "textGrain B=m=2, beta=0.25",
    "textgrain_B4_b0.1": "textGrain B=m=4, beta=0.1",
    "textgrain_B4_b0.25": "textGrain B=m=4, beta=0.25",
    "textgrain_B4_b0.5": "textGrain B=m=4, beta=0.5",
    "textgrain_B8_b0.25": "textGrain B=m=8, beta=0.25",
    "textgrain_B8_b0.5": "textGrain B=m=8, beta=0.5",
    "textgrain_B8_b0.75": "textGrain B=m=8, beta=0.75",
    "textgrain_B8_b0.25_L1": "textGrain B=m=8, beta=0.25, L=1",
    "textgrain_B8_b0.25_L4": "textGrain B=m=8, beta=0.25, L=4",
    "textgrain_B8_b0.25_start": "textGrain B=m=8, beta=0.25, unmarked start",
}
DIVERSITY = ["none", "gumbel", "greenlist_d2", "textgrain_B4_b0.25", "textgrain_B8_b0.25", "textgrain_B8_b0.5", "textgrain_B8_b0.25_start"]
LABELS = {"none": "no watermark", **CONFIGS}


def available(name: str) -> bool:
    return (RESULTS / f"{name}.jsonl").exists() and (RESULTS / f"{name}.scores.json").exists()


def records_of(name: str) -> list[dict]:
    return [json.loads(line) for line in (RESULTS / f"{name}.jsonl").open()]


def scores_of(name: str) -> list[dict]:
    return json.loads((RESULTS / f"{name}.scores.json").read_text())


def pvalue(times: list[int], scores: list[float], detector: str, n: int = 10**9) -> float:
    """Takes the scored positions and scores of a text and returns the p-value using its first n tokens."""
    y = np.array([s for t, s in zip(times, scores) if t < n])
    if len(y) == 0:
        return 1.0
    if detector == "greenlist":
        return float(binom.sf(y.sum() - 1, len(y), 0.25))
    return float(gamma.sf(y.sum(), a=len(y)))


def generated_key(scores: list[dict]) -> tuple[str, str]:
    (key,) = [k for k in scores[0]["scores"] if k.startswith("generated/")]
    return key, key.split("/")[1]


def power_table() -> dict:
    """Returns, for each watermark, its detection rate at each length (all texts, and texts not past their natural end) and its removed entropy."""
    table = {}
    for name in CONFIGS:
        if not available(name):
            continue
        records, scores = records_of(name), scores_of(name)
        key, detector = generated_key(scores)
        natural = [natural_length(r["stop_probs"]) for r in records]
        row = {"all": [], "filtered": [], "kept": []}
        for n in LENGTHS:
            ps = [pvalue(s["scores"][key]["t"], s["scores"][key]["y"], detector, n) for s in scores]
            kept = [p for p, length in zip(ps, natural) if length >= n]
            row["all"].append(detection_rate(ps, ALPHA))
            row["filtered"].append(detection_rate(kept, ALPHA))
            row["kept"].append(len(kept))
        removed = [r["removed_fraction"] for r in records if r["removed_fraction"] is not None]
        entropy = np.mean([np.mean(r["entropies"]) for r in records])
        row["removed"] = float(np.mean(removed)) if removed else None
        row["removed_nats"] = row["removed"] * entropy if removed else None
        y = np.concatenate([s["scores"][key]["y"] for s in scores])
        row["signal"] = float(y.mean() - (0.25 if detector == "greenlist" else 1.0))
        table[name] = row
    return table


def signal_by_entropy(names: list[str], edges: np.ndarray) -> dict[str, list[float]]:
    """Returns, for each watermark, the mean signal per scored position (Y - 1) in each bin of next-token entropy."""
    out = {}
    for name in names:
        if not available(name) or name == "greenlist_d2":
            continue
        records, scores = records_of(name), scores_of(name)
        key, _ = generated_key(scores)
        h, y = [], []
        for r, s in zip(records, scores):
            for t, value in zip(s["scores"][key]["t"], s["scores"][key]["y"]):
                if t < len(r["entropies"]):
                    h.append(r["entropies"][t])
                    y.append(value - 1)
        h, y = np.array(h), np.array(y)
        bins = np.digitize(h, edges) - 1
        out[name] = [float(y[bins == b].mean()) if (bins == b).sum() > 50 else np.nan for b in range(len(edges) - 1)]
    return out


def null_table(name: str, n: int = 400) -> dict[str, np.ndarray]:
    """Returns, for each detector, the p-values of the unwatermarked texts of a results file."""
    scores = scores_of(name)
    return {
        key: np.array([pvalue(s["scores"][key]["t"], s["scores"][key]["y"], key.split("/")[1], n) for s in scores])
        for key in scores[0]["scores"]
        if s_has(scores, key)
    }


def s_has(scores: list[dict], key: str) -> bool:
    return all(len(s["scores"][key]["y"]) > 0 for s in scores)


def ngrams(words: list[str], n: int) -> Counter:
    return Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def bleu(candidate: list[str], references: list[list[str]]) -> float:
    """Takes a tokenized text and tokenized references and returns its smoothed BLEU-4 score against them."""
    precisions = []
    for n in range(1, 5):
        counts = ngrams(candidate, n)
        best = Counter()
        for ref in references:
            best |= ngrams(ref, n)
        matched = sum(min(c, best[g]) for g, c in counts.items())
        precisions.append((matched + 1) / (sum(counts.values()) + 1))
    closest = min((abs(len(r) - len(candidate)), len(r)) for r in references)[1]
    brevity = 1.0 if len(candidate) > closest else np.exp(1 - closest / max(len(candidate), 1))
    return float(brevity * np.exp(np.mean(np.log(precisions))))


def diversity(name: str) -> dict:
    """Returns, for the answers to repeated questions, the mean number of distinct answers and the mean Self-BLEU per question."""
    groups: dict[int, list[str]] = {}
    for r in records_of(f"questions_{name}"):
        groups.setdefault(r["prompt_index"], []).append(r["text"])
    distinct, self_bleu = [], []
    for texts in groups.values():
        distinct.append(len(set(texts)))
        tokenized = [t.lower().split() for t in texts]
        self_bleu.append(np.mean([bleu(c, tokenized[:i] + tokenized[i + 1 :]) for i, c in enumerate(tokenized)]))
    return {"distinct": float(np.mean(distinct)), "self_bleu": float(np.mean(self_bleu)), "samples": len(next(iter(groups.values())))}


def perplexities(name: str) -> dict:
    data = json.loads((RESULTS / f"{name}.ppl.json").read_text())
    return {k: float(np.median([p for p in v if p is not None])) for k, v in data.items() if v}


def robustness(name: str) -> dict[str, float]:
    """Returns the detection rate of the watermarked texts after each attack."""
    path = RESULTS / f"{name}.attacks.scores.json"
    data = json.loads(path.read_text())
    detector = data[0]["detector"].split("_")[0]
    return {
        attack: detection_rate([pvalue(d["scores"][attack]["t"], d["scores"][attack]["y"], detector) for d in data], ALPHA)
        for attack in data[0]["scores"]
    }


def write_examples(path: Path) -> None:
    """Writes, for each showcase question, the first answer of each method to a Markdown file."""
    lines = ["# Answers to the same questions, with and without watermark", ""]
    answers = {name: records_of(f"questions_{name}") for name in DIVERSITY if (RESULTS / f"questions_{name}.jsonl").exists()}
    if not answers:
        return
    for q in range(4):
        question = next(r["prefix"] for r in next(iter(answers.values())) if r["prompt_index"] == q)
        lines += [f"## {question}", ""]
        for name, records in answers.items():
            first = [r for r in records if r["prompt_index"] == q][:2]
            lines += [f"**{LABELS[name]}**", ""] + [f"> {r['text'].strip()}".replace("\n", "\n> ") + "\n" for r in first]
    for name in ["gumbel", "greenlist_d2", "textgrain_B8_b0.25"]:
        attack_path = RESULTS / f"{name}.attacks.json"
        if attack_path.exists():
            first = json.loads(attack_path.read_text())[0]
            original = records_of(name)[first["index"]]["text"]
            lines += [f"## Attacks on one {LABELS[name]} text", "", f"**original**", "", "> " + original.strip()[:600].replace("\n", " "), ""]
            for attack, text in first["texts"].items():
                lines += [f"**{attack}**", "", "> " + text.strip()[:600].replace("\n", " "), ""]
            break
    path.write_text("\n".join(lines))


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    table = power_table()

    print(f"1. detection rate at alpha = {ALPHA} (all texts | texts not past their natural end)")
    print(f"{'':36}" + "".join(f"{n:>13}" for n in LENGTHS) + "  removed  signal/pos  signal/nat")
    for name, row in table.items():
        cells = "".join(f"{a:6.2f} |{f:5.2f}" for a, f in zip(row["all"], row["filtered"]))
        removed = f"{row['removed']:7.3f}" if row["removed"] is not None else "      -"
        efficiency = f"{row['signal'] / row['removed_nats']:11.2f}" if row["removed_nats"] else "          -"
        print(f"{CONFIGS[name]:36}{cells}  {removed}  {row['signal']:10.3f}{efficiency}")
    print("texts kept by the natural-end filter:", dict(zip(LENGTHS, table["gumbel"]["kept"])))

    print("\n2. false positives (p-values of unwatermarked texts, should be uniform)")
    for source in ["none", "human"]:
        if not available(source):
            continue
        for key, ps in null_table(source).items():
            print(f"{source + ':' + key:34} n = {len(ps):4d}  below 0.01: {np.mean(ps < 0.01):.3f}  below 0.05: {np.mean(ps < 0.05):.3f}  KS p-value: {kstest(ps, 'uniform').pvalue:.3f}")

    if (RESULTS / "none.ppl.json").exists():
        print("\n3. median perplexity under Qwen2.5-3B")
        print(f"{'human continuations':36}{perplexities('none')['human']:8.2f}")
        for name in ["none"] + list(CONFIGS):
            if (RESULTS / f"{name}.ppl.json").exists():
                print(f"{LABELS[name]:36}{perplexities(name)['generated']:8.2f}")

    div = {name: diversity(name) for name in DIVERSITY if (RESULTS / f"questions_{name}.jsonl").exists()}
    if div:
        print("\n4. diversity of 10 answers to the same question (30 questions)")
        for name, d in div.items():
            print(f"{LABELS[name]:36}distinct answers: {d['distinct']:5.2f} / {d['samples']}   Self-BLEU: {d['self_bleu']:.3f}")

    attacked = [name for name in CONFIGS if (RESULTS / f"{name}.attacks.scores.json").exists()]
    if attacked:
        print(f"\n5. detection rate after attacks (alpha = {ALPHA}, 100 texts)")
        for name in attacked:
            print(f"{CONFIGS[name]:36}" + "  ".join(f"{a}: {r:.2f}" for a, r in robustness(name).items()))

    panels = {
        "methods": ["gumbel", "greenlist_d2", "textgrain_B4_b0.25", "textgrain_B8_b0.25"],
        "textGrain settings": ["textgrain_B2_b0.25", "textgrain_B4_b0.1", "textgrain_B4_b0.25", "textgrain_B8_b0.25", "textgrain_B8_b0.5", "textgrain_B8_b0.25_L1"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (title, names) in zip(axes, panels.items()):
        for name in names:
            if name in table:
                ax.plot(LENGTHS, table[name]["all"], marker="o", label=CONFIGS[name])
        ax.set_xscale("log")
        ax.set_xticks(LENGTHS, [str(n) for n in LENGTHS])
        ax.set_xlabel("tokens read")
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"detection rate (alpha = {ALPHA})")
    fig.tight_layout()
    fig.savefig(FIGURES / "power_vs_length.png", dpi=150)
    plt.close(fig)

    edges = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 12])
    signal = signal_by_entropy(["gumbel", "textgrain_B4_b0.25", "textgrain_B8_b0.25", "textgrain_B8_b0.5"], edges)
    fig, ax = plt.subplots(figsize=(6, 4))
    centers = (edges[:-1] + edges[1:]) / 2
    for name, values in signal.items():
        ax.plot(centers, values, marker="o", label=CONFIGS[name])
    ax.set_xlabel("entropy of the next-token distribution H(P) (nats)")
    ax.set_ylabel("mean signal per position, E[Y] - 1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "signal_vs_entropy.png", dpi=150)
    plt.close(fig)

    if div:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, d in div.items():
            power = table[name]["all"][LENGTHS.index(50)] if name in table else 0.0
            ax.scatter(1 - d["self_bleu"], power, s=70, label=LABELS[name])
        ax.set_xlim(0, 1)
        ax.set_xlabel("diversity of 10 answers to the same question (1 - Self-BLEU)")
        ax.set_ylabel(f"detection rate at 50 tokens (alpha = {ALPHA})")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(FIGURES / "power_vs_diversity.png", dpi=150)
        plt.close(fig)

    if available("human"):
        nulls = null_table("human")
        fig, axes = plt.subplots(1, len(nulls), figsize=(3.2 * len(nulls), 3), sharey=True)
        for ax, (key, ps) in zip(axes, nulls.items()):
            ax.hist(ps, bins=20, range=(0, 1), density=True)
            ax.axhline(1, color="k", linestyle="--")
            ax.set_title(key.split("/")[1], fontsize=9)
            ax.set_xlabel("p-value")
        fig.tight_layout()
        fig.savefig(FIGURES / "null_pvalues.png", dpi=150)
        plt.close(fig)

    write_examples(FIGURES / "examples.md")


if __name__ == "__main__":
    main()
