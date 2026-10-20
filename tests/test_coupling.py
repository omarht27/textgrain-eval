"""Checks the block coupling on the toy example of the project brief and on random distributions."""

import numpy as np

from textgrain.coupling import block_probs, couple, entropy, kl_from_independence, sinkhorn, standardized_gumbel, token_probs
from textgrain.keys import keyed_draws, keyed_rng

KEY = bytes.fromhex("00112233445566778899aabbccddeeff" * 2)
P = np.array([0.35, 0.25, 0.15, 0.10, 0.10, 0.05])
EXPECTED_PI = np.array([[0.121, 0.028, 0.001], [0.123, 0.297, 0.030], [0.090, 0.009, 0.302]])


def toy() -> tuple:
    blocks, uniforms, column = keyed_draws(keyed_rng(KEY, [21, 22]), 6, 3, 3)
    rho = block_probs(P, blocks, 3)
    return blocks, rho, standardized_gumbel(uniforms), column


def test_toy_coupling() -> None:
    blocks, rho, gains, column = toy()
    result = couple(rho, gains, entropy(P), beta=0.25)
    assert np.allclose(rho, [0.15, 0.45, 0.40])
    assert abs(result.lam - 0.628) < 2e-3
    assert abs(result.beta - 0.25) < 1e-3
    assert np.allclose(result.pi, EXPECTED_PI, atol=1e-3)
    assert np.allclose(3 * result.pi[:, column], [0.004, 0.091, 0.905], atol=1e-3)


def test_marginals_and_average_over_columns() -> None:
    blocks, rho, gains, column = toy()
    pi = couple(rho, gains, entropy(P), beta=0.25).pi
    assert np.allclose(pi.sum(axis=1), rho, atol=1e-12)
    assert np.allclose(pi.sum(axis=0), 1 / 3, atol=1e-12)
    average = np.mean([token_probs(P, blocks, rho, pi, j) for j in range(3)], axis=0)
    assert np.allclose(average, P, atol=1e-12)


def test_kl_decreases_with_lambda() -> None:
    _, rho, gains, _ = toy()
    kls = [kl_from_independence(sinkhorn(rho, gains, lam), rho) for lam in np.logspace(-2, 2, 20)]
    assert all(a >= b - 1e-9 for a, b in zip(kls, kls[1:]))


def test_zero_budget_and_empty_blocks() -> None:
    rng = np.random.default_rng(0)
    rho = np.array([0.6, 0.0, 0.4, 0.0])
    gains = standardized_gumbel(rng.random((4, 4)))
    assert couple(rho, gains, 1.0, beta=0.0).beta == 0.0
    result = couple(rho, gains, 1.0, beta=0.3)
    assert np.allclose(result.pi[[1, 3]], 0)
    assert np.allclose(result.pi.sum(axis=1), rho, atol=1e-12)


def test_random_distributions_hit_the_budget() -> None:
    rng = np.random.default_rng(1)
    for beta in [0.05, 0.1, 0.25]:
        for _ in range(10):
            probs = rng.dirichlet(np.full(1000, 0.1))
            blocks = rng.integers(0, 4, size=1000)
            rho = block_probs(probs, blocks, 4)
            result = couple(rho, standardized_gumbel(rng.random((4, 4))), entropy(probs), beta=beta)
            target = min(beta, min(entropy(rho), np.log(4)) / entropy(probs))
            assert result.beta <= target * (1 + 2e-3)
            if result.lam > 1.01e-2:
                assert abs(result.beta - target) < 2e-3 * target
