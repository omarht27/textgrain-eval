"""Helpers to analyse generated texts."""

import numpy as np


def natural_length(stop_probs: list[float], threshold: float = 0.5) -> int:
    """Takes the stopping probability at each step and returns the first step where it exceeds threshold, or the full length if it never does."""
    above = np.flatnonzero(np.array(stop_probs) > threshold)
    return int(above[0]) if len(above) else len(stop_probs)


def detection_rate(pvalues: list[float], alpha: float = 0.01) -> float:
    """Takes p-values and returns the fraction below alpha."""
    return float(np.mean(np.array(pvalues) < alpha)) if len(pvalues) else float("nan")
