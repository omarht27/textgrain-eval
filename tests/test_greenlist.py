"""Checks the green-list generator and detector on synthetic distributions."""

import numpy as np

from textgrain.detect import binomial_pvalue, greenlist_hits
from textgrain.greenlist import green_mask, greenlist_chooser
from textgrain.keys import new_key

VOCAB_SIZE = 300


def test_green_fraction() -> None:
    mask = green_mask(new_key(), [1, 2], 100_000, 0.25)
    assert abs(mask.mean() - 0.25) < 0.01


def test_detects_own_key_only() -> None:
    key, rng = new_key(), np.random.default_rng(0)
    choose = greenlist_chooser(key, 2, gamma=0.25, delta=2.0, seed=0)
    tokens: list[int] = []
    for _ in range(150):
        tokens.append(choose(rng.dirichlet(np.full(VOCAB_SIZE, 0.05)), tokens))
    assert binomial_pvalue(greenlist_hits(tokens, key, 2, VOCAB_SIZE, 0.25), 0.25) < 1e-6
    assert binomial_pvalue(greenlist_hits(tokens, new_key(), 2, VOCAB_SIZE, 0.25), 0.25) > 1e-3


def test_pvalues_under_h0() -> None:
    rng = np.random.default_rng(1)
    pvalues = np.array([binomial_pvalue(rng.random(200) < 0.25, 0.25) for _ in range(5000)])
    assert (pvalues < 0.05).mean() <= 0.05 + 0.01
