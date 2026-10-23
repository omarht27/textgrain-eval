"""Detection scores and p-values."""

from collections.abc import Iterator

import numpy as np
from scipy.stats import binom, gamma

from textgrain.greenlist import green_mask
from textgrain.keys import context_window, keyed_draws, keyed_rng


def scored_positions(tokens: list[int], window: int, min_context: int = 0) -> Iterator[tuple[list[int], int]]:
    """Takes token ids and yields (context, token) at the first occurrence of each context."""
    for t in scored_indices(tokens, window, min_context):
        yield context_window(tokens, t, window), tokens[t]


def scored_indices(tokens: list[int], window: int, min_context: int = 0) -> list[int]:
    """Takes token ids and returns the positions t scored by the detectors: the first occurrence of each context with at least min_context tokens."""
    seen: set[tuple[int, ...]] = set()
    indices = []
    for t in range(min(min_context, len(tokens)), len(tokens)):
        context = tuple(context_window(tokens, t, window))
        if context not in seen:
            seen.add(context)
            indices.append(t)
    return indices


def gumbel_scores(tokens: list[int], key: bytes, window: int, vocab_size: int, min_context: int = 0) -> np.ndarray:
    """Takes token ids and returns Y = -ln(1 - U_token) at each scored position, for the Gumbel-max watermark."""
    uniforms = [keyed_rng(key, context).random(vocab_size)[token] for context, token in scored_positions(tokens, window, min_context)]
    return -np.log1p(-np.array(uniforms))


def textgrain_scores(tokens: list[int], key: bytes, window: int, vocab_size: int, n_blocks: int, n_columns: int, min_context: int = 0) -> np.ndarray:
    """Takes token ids and returns Y = -ln(1 - U[block of token, J]) at each scored position, for the textGrain watermark."""
    uniforms = []
    for context, token in scored_positions(tokens, window, min_context):
        blocks, table, column = keyed_draws(keyed_rng(key, context), vocab_size, n_blocks, n_columns)
        uniforms.append(table[blocks[token], column])
    return -np.log1p(-np.array(uniforms))


def gamma_pvalue(scores: np.ndarray) -> float:
    """Takes scores Y_1, ..., Y_n, i.i.d. Exp(1) under H0, and returns the p-value P(Gamma(n, 1) >= S_n)."""
    return float(gamma.sf(scores.sum(), a=len(scores))) if len(scores) else 1.0


def greenlist_hits(tokens: list[int], key: bytes, window: int, vocab_size: int, green_fraction: float, min_context: int = 0) -> np.ndarray:
    """Takes token ids and returns 1 for each scored token that is in its green list and 0 otherwise."""
    return np.array([green_mask(key, context, vocab_size, green_fraction)[token] for context, token in scored_positions(tokens, window, min_context)], dtype=int)


def binomial_pvalue(hits: np.ndarray, green_fraction: float) -> float:
    """Takes green hits, i.i.d. Bernoulli(gamma) under H0, and returns the p-value P(Binomial(n, gamma) >= number of hits)."""
    return float(binom.sf(hits.sum() - 1, len(hits), green_fraction)) if len(hits) else 1.0


def z_score(hits: np.ndarray, green_fraction: float) -> float:
    """Takes green hits and returns the z-score (hits - gamma n) / sqrt(n gamma (1 - gamma)) of Kirchenbauer et al."""
    n = len(hits)
    return float((hits.sum() - green_fraction * n) / np.sqrt(n * green_fraction * (1 - green_fraction)))
