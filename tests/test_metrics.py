import numpy as np

from attribution.evaluation.metrics import (
    classification_report,
    delta_accuracy,
    expected_calibration_error,
    selective_report,
)
from attribution.models.calibration import fit_temperature


def test_classification_report():
    r = classification_report([1, 1, 0, 0], [1, 0, 1, 0])
    assert r["accuracy"] == 0.5 and r["precision"] == 0.5 and r["recall"] == 0.5 and r["fpr"] == 0.5
    assert delta_accuracy(0.95, 0.80) == 0.95 - 0.80


def test_selective_report_coverage():
    r = selective_report([1, 0, 1], ["ai", "human", "uncertain"])
    assert r["coverage"] == 2 / 3 and r["accuracy"] == 1.0


def test_temperature_scaling_reduces_ece_for_overconfident_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    true_logits = rng.normal(0, 1.5, 5000) + (2 * y - 1) * 1.0
    p_true = 1 / (1 + np.exp(-true_logits))
    y = (rng.random(5000) < p_true).astype(int)
    overconfident = true_logits * 4
    t = fit_temperature(overconfident, y)
    assert 3 < t < 5
    before = expected_calibration_error(y, 1 / (1 + np.exp(-overconfident)))
    after = expected_calibration_error(y, 1 / (1 + np.exp(-overconfident / t)))
    assert after < before
