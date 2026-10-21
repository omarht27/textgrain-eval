"""Turns the results and scores files into tables and figures."""

import json
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
    "none": "no watermark",
    "gumbel": "Gumbel-max",
    "greenlist_d2": "green list (delta=2)",
    "textgrain_B4_b0.1": "textGrain B=m=4, beta=0.1",
    "textgrain_B4_b0.25": "textGrain B=m=4, beta=0.25",
    "textgrain_B4_b0.5": "textGrain B=m=4, beta=0.5",
    "textgrain_B2_b0.25": "textGrain B=m=2, beta=0.25",
    "textgrain_B8_b0.25": "textGrain B=m=8, beta=0.25",
    "textgrain_B8_b0.5": "textGrain B=m=8, beta=0.5",
    "textgrain_B8_b0.75": "textGrain B=m=8, beta=0.75",
}


def pvalue(times: list[int], scores: list[float], detector: str, n: int) -> float:
    """Takes the scored positions and scores of a text and returns the p-value using its first n tokens."""
    y = np.array([s for t, s in zip(times, scores) if t < n])
    if len(y) == 0:
        return 1.0
    if detector == "greenlist":
        return float(binom.sf(y.sum() - 1, len(y), 0.25))
    return float(gamma.sf(y.sum(), a=len(y)))


def load(name: str) -> tuple[list[dict], list[dict]]:
    """Takes a configuration name and returns its generated records and their scores."""
    records = [json.loads(line) for line in (RESULTS / f"{name}.jsonl").open()]
    scores = json.loads((RESULTS / f"{name}.scores.json").read_text())
    return records, scores


def power_table() -> dict[str, dict[str, list[float]]]:
    """Returns, for each watermark, the detection rate at each length, on all texts and on texts not past their natural end."""
    table = {}
    for name in CONFIGS:
        if name == "none":
            continue
        records, scores = load(name)
        (key,) = [k for k in scores[0]["scores"] if k.startswith("generated/")]
        detector = key.split("/")[1]
        natural = [natural_length(r["stop_probs"]) for r in records]
        all_rates, filtered_rates, counts = [], [], []
        for n in LENGTHS:
            ps = [pvalue(s["scores"][key]["t"], s["scores"][key]["y"], detector, n) for s in scores]
            kept = [p for p, length in zip(ps, natural) if length >= n]
            all_rates.append(detection_rate(ps, ALPHA))
            filtered_rates.append(detection_rate(kept, ALPHA))
            counts.append(len(kept))
        removed = [r["removed_fraction"] for r in records if r["removed_fraction"] is not None]
        table[name] = {"all": all_rates, "filtered": filtered_rates, "kept": counts, "removed": float(np.mean(removed)) if removed else None}
    return table


def null_table() -> dict[str, dict[str, list[float]]]:
    """Returns, for each detector, the p-values of the human and unwatermarked model texts at 400 tokens."""
    _, scores = load("none")
    out = {}
    for key in scores[0]["scores"]:
        source, detector = key.split("/")
        ps = [pvalue(s["scores"][key]["t"], s["scores"][key]["y"], detector, 400) for s in scores]
        out[key] = ps
    return out


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    table = power_table()
    print(f"detection rate at alpha = {ALPHA} (all texts | texts not past their natural end)")
    print(f"{'':32}" + "".join(f"{n:>13}" for n in LENGTHS) + "   removed")
    for name, row in table.items():
        cells = "".join(f"{a:6.2f} |{f:5.2f}" for a, f in zip(row["all"], row["filtered"]))
        removed = f"{row['removed']:.3f}" if row["removed"] is not None else "   -"
        print(f"{CONFIGS[name]:32}{cells}   {removed}")
    print("texts kept by the natural-end filter:", dict(zip(LENGTHS, next(iter(table.values()))["kept"])))

    print("\nnull p-values at 400 tokens (should be uniform)")
    nulls = null_table()
    for key, ps in nulls.items():
        ps = np.array(ps)
        print(f"{key:28} below 0.01: {np.mean(ps < 0.01):.3f}   below 0.05: {np.mean(ps < 0.05):.3f}   KS p-value: {kstest(ps, 'uniform').pvalue:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, kind in zip(axes, ["all", "filtered"]):
        for name, row in table.items():
            ax.plot(LENGTHS, row[kind], marker="o", label=CONFIGS[name])
        ax.set_xscale("log")
        ax.set_xticks(LENGTHS, [str(n) for n in LENGTHS])
        ax.set_xlabel("tokens read")
        ax.set_title("all texts" if kind == "all" else "texts not past their natural end")
    axes[0].set_ylabel(f"detection rate (alpha = {ALPHA})")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "power_vs_length.png", dpi=150)

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
    for ax, (key, ps) in zip(axes.flat, nulls.items()):
        ax.hist(ps, bins=10, range=(0, 1), density=True)
        ax.axhline(1, color="k", linestyle="--")
        ax.set_title(key, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "null_pvalues.png", dpi=150)


if __name__ == "__main__":
    main()
