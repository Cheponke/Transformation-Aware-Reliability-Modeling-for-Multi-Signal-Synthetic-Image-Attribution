"""Paired statistical tests for comparing attribution systems on the same images."""
from __future__ import annotations

from math import comb

import numpy as np


def mcnemar_test(y_true, pred_a, pred_b) -> dict[str, float | int | None]:
    """Exact two-sided McNemar test for paired binary predictions.

    Args:
        y_true: Binary ground-truth labels (0=human, 1=AI).
        pred_a: Binary predictions from system A, aligned with y_true.
        pred_b: Binary predictions from system B, aligned with y_true.

    Returns:
        Discordant-pair counts, exact two-sided p-value, and each system's
        accuracy. Abstentions must be excluded consistently before calling.
    """
    y = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)

    if y.ndim != 1 or a.ndim != 1 or b.ndim != 1:
        raise ValueError("y_true, pred_a, and pred_b must be one-dimensional.")
    if not (len(y) == len(a) == len(b)):
        raise ValueError("y_true, pred_a, and pred_b must have equal lengths.")
    if len(y) == 0:
        raise ValueError("At least one paired prediction is required.")
    if not np.isin(y, [0, 1]).all() or not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("All labels and predictions must be binary (0 or 1).")

    correct_a = a == y
    correct_b = b == y

    # b_a: A correct / B wrong; b_b: A wrong / B correct.
    b_a = int(np.sum(correct_a & ~correct_b))
    b_b = int(np.sum(~correct_a & correct_b))
    discordant = b_a + b_b

    if discordant == 0:
        p_value = 1.0
    else:
        lower_tail = sum(comb(discordant, k) for k in range(min(b_a, b_b) + 1))
        p_value = min(1.0, 2.0 * lower_tail / (2 ** discordant))

    return {
        "n": int(len(y)),
        "a_correct_b_wrong": b_a,
        "a_wrong_b_correct": b_b,
        "accuracy_a": float(np.mean(correct_a)),
        "accuracy_b": float(np.mean(correct_b)),
        "accuracy_difference_a_minus_b": float(np.mean(correct_a) - np.mean(correct_b)),
        "mcnemar_exact_p": float(p_value),
    }
