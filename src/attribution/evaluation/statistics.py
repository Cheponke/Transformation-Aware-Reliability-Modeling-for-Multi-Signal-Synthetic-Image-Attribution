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
