"""Green-list watermark (Kirchenbauer et al., 2023)."""

import numpy as np

from textgrain.keys import context_window, keyed_rng
from textgrain.sampling import TokenChooser, sample


def green_mask(key: bytes, context: list[int], vocab_size: int, gamma: float) -> np.ndarray:
    """Takes the key, a context window, the vocabulary size and the green fraction and returns a boolean mask of green tokens (V,)."""
    return keyed_rng(key, context).random(vocab_size) < gamma


def greenlist_chooser(key: bytes, window: int, gamma: float, delta: float, seed: int, mask_repeated: bool = True) -> TokenChooser:
    """Takes the key, the context window length, the green fraction gamma, the logit bias delta and a seed and returns a green-list token chooser."""
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()

    def choose(probs: np.ndarray, response: list[int]) -> int:
        if not response:
            seen.clear()
        context = tuple(context_window(response, len(response), window))
        if mask_repeated and context in seen:
            return sample(probs, rng)
        seen.add(context)
        biased = probs * np.where(green_mask(key, list(context), len(probs), gamma), np.exp(delta), 1.0)
        return sample(biased / biased.sum(), rng)

    return choose
