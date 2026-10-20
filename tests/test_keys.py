"""Checks the keyed draws on the toy example of the project brief."""

import numpy as np

from textgrain.keys import context_window, keyed_draws, keyed_rng, position_seed

KEY = bytes.fromhex("00112233445566778899aabbccddeeff" * 2)
CONTEXT = [21, 22]
VOCAB = ["lait", "eau", "jus", "café", "thé", "soda"]


def test_hmac_prefix() -> None:
    assert position_seed(KEY, CONTEXT).hex().startswith("6ea6085d")


def test_toy_draws() -> None:
    blocks, uniforms, column = keyed_draws(keyed_rng(KEY, CONTEXT), vocab_size=6, n_blocks=3, n_columns=3)
    members = [{VOCAB[w] for w in np.flatnonzero(blocks == b)} for b in range(3)]
    assert members == [{"café", "soda"}, {"lait", "thé"}, {"jus", "eau"}]
    assert uniforms.shape == (3, 3)
    assert column == 2


def test_draws_are_reproducible() -> None:
    first = keyed_draws(keyed_rng(KEY, CONTEXT), 1000, 4, 4)
    second = keyed_draws(keyed_rng(KEY, CONTEXT), 1000, 4, 4)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1]) and first[2] == second[2]


def test_context_window() -> None:
    assert context_window([5, 6, 7, 8], t=3, window=2) == [6, 7]
    assert context_window([5, 6, 7, 8], t=1, window=2) == [5]
    assert context_window([5, 6, 7, 8], t=0, window=2) == []
