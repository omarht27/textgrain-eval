"""Checks that Gumbel-max sampling follows P when the uniforms are random."""

import numpy as np

from textgrain.detect import gamma_pvalue
from textgrain.gumbel import gumbel_argmax

P = np.array([0.35, 0.25, 0.15, 0.10, 0.10, 0.05])


def test_gumbel_argmax_follows_p() -> None:
    rng = np.random.default_rng(0)
    draws = [gumbel_argmax(P, rng.random(len(P))) for _ in range(50_000)]
    frequencies = np.bincount(draws, minlength=len(P)) / len(draws)
    assert np.abs(frequencies - P).max() < 0.01


def test_gumbel_argmax_ignores_zero_probabilities() -> None:
    rng = np.random.default_rng(1)
    probs = np.array([0.0, 0.5, 0.0, 0.5])
    assert all(gumbel_argmax(probs, rng.random(4)) in (1, 3) for _ in range(1000))


def test_pvalues_are_uniform_under_h0() -> None:
    rng = np.random.default_rng(2)
    pvalues = np.array([gamma_pvalue(rng.exponential(size=100)) for _ in range(5000)])
    assert abs((pvalues < 0.05).mean() - 0.05) < 0.01
