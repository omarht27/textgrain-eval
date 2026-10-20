"""Checks that the textGrain generator and detector agree, on synthetic distributions."""

import numpy as np

from textgrain.block_ot import removed_fraction, textgrain_chooser
from textgrain.detect import gamma_pvalue, textgrain_scores
from textgrain.keys import new_key

VOCAB_SIZE = 300


def generate_synthetic(choose, length: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    response: list[int] = []
    for _ in range(length):
        response.append(choose(rng.dirichlet(np.full(VOCAB_SIZE, 0.05)), response))
    return response


def test_detects_own_key_only() -> None:
    key, log = new_key(), []
    tokens = generate_synthetic(textgrain_chooser(key, 2, 4, 4, beta=0.25, seed=0, log=log), 150, seed=0)
    assert gamma_pvalue(textgrain_scores(tokens, key, 2, VOCAB_SIZE, 4, 4)) < 1e-6
    assert gamma_pvalue(textgrain_scores(tokens, new_key(), 2, VOCAB_SIZE, 4, 4)) > 1e-3
    assert 0 < removed_fraction(log) <= 0.25 + 1e-3


def test_zero_budget_leaves_no_signal() -> None:
    key = new_key()
    pvalues = []
    for seed in range(20):
        tokens = generate_synthetic(textgrain_chooser(key, 2, 4, 4, beta=0.0, seed=seed), 100, seed=seed)
        pvalues.append(gamma_pvalue(textgrain_scores(tokens, key, 2, VOCAB_SIZE, 4, 4)))
    assert min(pvalues) > 1e-4
