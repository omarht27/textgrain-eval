"""Checks the analysis helpers."""

from textgrain.analysis import detection_rate, natural_length


def test_natural_length() -> None:
    assert natural_length([0.1, 0.2, 0.7, 0.9]) == 2
    assert natural_length([0.1, 0.2]) == 2


def test_detection_rate() -> None:
    assert detection_rate([0.001, 0.5, 0.009, 0.2]) == 0.5
