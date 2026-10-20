"""textGrain: watermarking with block optimal transport and an entropy budget."""

import numpy as np

from textgrain.coupling import block_probs, couple, entropy, standardized_gumbel
from textgrain.keys import context_window, keyed_draws, keyed_rng
from textgrain.sampling import TokenChooser, sample


def textgrain_chooser(
    key: bytes,
    window: int,
    n_blocks: int,
    n_columns: int,
    beta: float,
    seed: int,
    mask_repeated: bool = True,
    log: list[tuple[float, float]] | None = None,
) -> TokenChooser:
    """Takes the key, the watermark parameters and a seed for the free generator and returns a textGrain token chooser; if given, log receives (H(P), removed fraction) at each watermarked position."""
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()

    def choose(probs: np.ndarray, response: list[int]) -> int:
        if not response:
            seen.clear()
        context = tuple(context_window(response, len(response), window))
        if mask_repeated and context in seen:
            return sample(probs, rng)
        seen.add(context)

        blocks, uniforms, column = keyed_draws(keyed_rng(key, list(context)), len(probs), n_blocks, n_columns)
        rho = block_probs(probs, blocks, n_blocks)
        token_entropy = entropy(probs)
        coupling = couple(rho, standardized_gumbel(uniforms), token_entropy, beta)
        if log is not None:
            log.append((token_entropy, coupling.beta))

        block = sample(n_columns * coupling.pi[:, column], rng)
        return sample(np.where(blocks == block, probs, 0.0), rng)

    return choose


def removed_fraction(log: list[tuple[float, float]]) -> float:
    """Takes the log of a generation and returns the entropy removed over the whole text, as a fraction of the total entropy."""
    entropies, fractions = np.array(log).T
    total = entropies.sum()
    return float((entropies * fractions).sum() / total) if total > 0 else 0.0
