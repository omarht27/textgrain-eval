"""Secret key, per-position seeds and keyed random draws."""

import hashlib
import hmac
import secrets
from pathlib import Path

import numpy as np

KEY_BYTES = 32


def new_key() -> bytes:
    """Returns a fresh 32-byte secret key drawn from the operating system."""
    return secrets.token_bytes(KEY_BYTES)


def save_key(key: bytes, path: str | Path) -> None:
    """Takes a key and writes it as hexadecimal text to path."""
    Path(path).write_text(key.hex())


def load_key(path: str | Path) -> bytes:
    """Takes the path of a key file and returns the key."""
    return bytes.fromhex(Path(path).read_text().strip())


def context_window(response: list[int], t: int, window: int) -> list[int]:
    """Takes the response, a position t and a window length L and returns the (up to) L tokens before t."""
    return response[max(0, t - window) : t]


def position_seed(key: bytes, context: list[int]) -> bytes:
    """Takes the key and a context window and returns HMAC-SHA256(key, context) on 32 bytes, each token id encoded on 4 bytes big-endian."""
    message = b"".join(token.to_bytes(4, "big") for token in context)
    return hmac.new(key, message, hashlib.sha256).digest()


def keyed_rng(key: bytes, context: list[int]) -> np.random.Generator:
    """Takes the key and a context window and returns a NumPy generator seeded with their HMAC."""
    return np.random.default_rng(int.from_bytes(position_seed(key, context), "big"))


def keyed_draws(rng: np.random.Generator, vocab_size: int, n_blocks: int, n_columns: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Takes a keyed generator and returns, drawn in this order, the block of each token (V,), the uniform table U (B, m) and the column J."""
    permutation = rng.permutation(vocab_size)
    blocks = np.empty(vocab_size, dtype=np.int64)
    blocks[permutation] = np.arange(vocab_size) * n_blocks // vocab_size
    uniforms = rng.random((n_blocks, n_columns))
    column = int(rng.integers(0, n_columns))
    return blocks, uniforms, column
