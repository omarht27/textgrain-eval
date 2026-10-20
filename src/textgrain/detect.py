"""Detection scores and p-values."""

from collections.abc import Iterator

import numpy as np
from scipy.stats import binom, gamma

from textgrain.greenlist import green_mask
from textgrain.keys import context_window, keyed_draws, keyed_rng


def scored_positions(tokens: list[int], window: int) -> Iterator[tuple[list[int], int]]:
    """Takes token ids and yields (context, token) at the first occurrence of each context."""
    seen: set[tuple[int, ...]] = set()
    for t, token in enumerate(tokens):
        context = context_window(tokens, t, window)
        if tuple(context) not in seen:
            seen.add(tuple(context))
            yield context, token


def gumbel_scores(tokens: list[int], key: bytes, window: int, vocab_size: int) -> np.ndarray:
    """Takes token ids and returns Y = -ln(1 - U_token) at each scored position, for the Gumbel-max watermark."""
    uniforms = [keyed_rng(key, context).random(vocab_size)[token] for context, token in scored_positions(tokens, window)]
    return -np.log1p(-np.array(uniforms))


def textgrain_scores(tokens: list[int], key: bytes, window: int, vocab_size: int, n_blocks: int, n_columns: int) -> np.ndarray:
    """Takes token ids and returns Y = -ln(1 - U[block of token, J]) at each scored position, for the textGrain watermark."""
    uniforms = []
    for context, token in scored_positions(tokens, window):
        blocks, table, column = keyed_draws(keyed_rng(key, context), vocab_size, n_blocks, n_columns)
        uniforms.append(table[blocks[token], column])
    return -np.log1p(-np.array(uniforms))


def gamma_pvalue(scores: np.ndarray) -> float:
    """Takes scores Y_1, ..., Y_n, i.i.d. Exp(1) under H0, and returns the p-value P(Gamma(n, 1) >= S_n)."""
    return float(gamma.sf(scores.sum(), a=len(scores))) if len(scores) else 1.0


def greenlist_hits(tokens: list[int], key: bytes, window: int, vocab_size: int, green_fraction: float) -> np.ndarray:
    """Takes token ids and returns 1 for each scored token that is in its green list and 0 otherwise."""
    return np.array([green_mask(key, context, vocab_size, green_fraction)[token] for context, token in scored_positions(tokens, window)], dtype=int)


def binomial_pvalue(hits: np.ndarray, green_fraction: float) -> float:
    """Takes green hits, i.i.d. Bernoulli(gamma) under H0, and returns the p-value P(Binomial(n, gamma) >= number of hits)."""
    return float(binom.sf(hits.sum() - 1, len(hits), green_fraction)) if len(hits) else 1.0


def z_score(hits: np.ndarray, green_fraction: float) -> float:
    """Takes green hits and returns the z-score (hits - gamma n) / sqrt(n gamma (1 - gamma)) of Kirchenbauer et al."""
    n = len(hits)
    return float((hits.sum() - green_fraction * n) / np.sqrt(n * green_fraction * (1 - green_fraction)))
