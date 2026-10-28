import pytest
from attribution.evaluation.statistics import paired_f1_test

def test_paired_f1_test_returns_expected_fields():
  result = paired_f1_test(
      [0.90,0.88,0.92,0.91,0.89],
      [0.84,0.82,0.86,0.85,0.83],
  )
  assert result["n_pairs"] == 5
  assert result["mean_f1_difference"] == pytest.approx(0.06)
  assert result["p_value"] < 0.05
  assert result["significant"] is True

def test_mismatched_pairs_raise_error():
  with pytest.raises(ValueError):
    paired_f1_test([0.9,0.8],[0.7])

def test_one_pair_raises_error():
  with pytest.raises(ValueError):
    paired_f1_test([0.9],[0.8])

def test_scores_outside_valid_range_raise_error():
  with pytest.raises(ValueError):
    paired_f1_test([1.2, 0.8],[0.7, 0.6])