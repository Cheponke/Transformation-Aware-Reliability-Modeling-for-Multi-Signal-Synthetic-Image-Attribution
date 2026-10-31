"""Paired statistical tests for comparing attribution systems on the same images."""
import numpy as np
from scipy.stats import ttest_rel

def paired_f1_test(
    multi_signal_f1: list[float],
    cnn_only_f1: list[float],
) -> dict[str, float | int | bool]:
  multi = np.asarray(multi_signal_f1,dtype=float)
  cnn = np.asarray(cnn_only_f1, dtype=float)

  if multi.ndim != 1 or cnn.ndim != 1:
    raise ValueError("F1 scores must be one dimensional.")
  if len(multi) != len(cnn):
    raise ValueError("Both systems must have the same number of paired scores.")
  if len(multi) < 2:
    raise ValueError("At least two paired observations are required.")
  if not np.isfinite(multi).all() or not np.isfinite(cnn).all():
    raise ValueError("F1 scores must be finite numers.")
  if ((multi < 0) | (multi >1)).any() or ((cnn < 0) | (cnn > 1)).any():
    raise ValueError("F1 scores must be between 0 and 1.")

  differences = multi - cnn
  test = ttest_rel(multi, cnn)
  p_value=float(test.pvalue)

  return {
      "n_pairs": len(multi),
      "mean_multi_signal_f1": float(np.mean(multi)),
      "mean_f1_difference": float(np.mean(differences)),
      "t_statistic": float(test.statistic),
      "p_value": p_value,
      "alpha": 0.05,
      "significant": bool(p_value < 0.05),
  }


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjusted p-values (None entries are skipped and stay None).

    Controls the chance of any false positive across a family of tests, e.g. one
    paired t-test per transformation condition and comparison.
    """
    indexed = sorted((p, i) for i, p in enumerate(p_values) if p is not None and np.isfinite(p))
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted
