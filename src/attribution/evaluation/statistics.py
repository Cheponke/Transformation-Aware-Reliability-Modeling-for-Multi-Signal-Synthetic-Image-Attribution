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


def corrected_paired_t_test(
    scores_a: list[float],
    scores_b: list[float],
    n_train: float,
    n_test: float,
) -> dict[str, float | int]:
    """Nadeau-Bengio corrected resampled t-test for k-fold cross-validation scores.

    The k training sets overlap, so the fold differences are positively
    correlated and the plain paired t-test understates their variance. The
    correction inflates the variance of the mean difference from s^2/k to
    (1/k + n_test/n_train) * s^2 (Nadeau & Bengio, 2003, Machine Learning 52).
    Returns the corrected t statistic and two-sided p-value (k - 1 degrees of freedom).
    """
    from scipy.stats import t as t_dist

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Both systems must have the same number of one-dimensional paired scores.")
    k = len(a)
    if k < 2:
        raise ValueError("At least two paired observations are required.")
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive.")
    d = a - b
    var = float(np.var(d, ddof=1))
    if var == 0:
        return {"n_pairs": k, "t_statistic": float("nan"), "p_value": float("nan"), "correction": 1 / k + n_test / n_train}
    t = float(np.mean(d)) / np.sqrt((1 / k + n_test / n_train) * var)
    return {"n_pairs": k, "t_statistic": t, "p_value": float(2 * t_dist.sf(abs(t), k - 1)),
            "correction": 1 / k + n_test / n_train}
