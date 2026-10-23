"""Gumbel-max watermark (Aaronson, 2023)."""

import numpy as np

from textgrain.keys import context_window, keyed_rng
from textgrain.sampling import TokenChooser, sample


def gumbel_argmax(probs: np.ndarray, uniforms: np.ndarray) -> int:
    """Takes a distribution and one uniform per token and returns argmax_w U_w^(1/P_w), which equals argmax_w ln P_w + Z_w."""
    with np.errstate(divide="ignore"):
        return int(np.argmax(np.log(uniforms) / probs))


def gumbel_chooser(key: bytes, window: int, seed: int, mask_repeated: bool = True, min_context: int = 0) -> TokenChooser:
    """Takes the key, the context window length and a seed for the free generator and returns a Gumbel-max token chooser."""
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()

    def choose(probs: np.ndarray, response: list[int]) -> int:
        if not response:
            seen.clear()
        context = tuple(context_window(response, len(response), window))
        if len(context) < min_context or (mask_repeated and context in seen):
            return sample(probs, rng)
        seen.add(context)
        return gumbel_argmax(probs, keyed_rng(key, list(context)).random(len(probs)))

    return choose
