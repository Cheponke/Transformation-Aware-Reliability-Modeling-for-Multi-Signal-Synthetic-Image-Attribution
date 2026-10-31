"""Cross-validation building blocks (manuscript 3.8.1): grouped folds, Holm correction, aggregation."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from attribution.data.datasets import Record
from attribution.data.eda import near_duplicate_groups
from attribution.data.splits import grouped_holdout, grouped_k_folds
from attribution.evaluation.statistics import holm_adjust


def _records(n_human, n_ai):
    return [Record(f"h{i}", 0, "s") for i in range(n_human)] + [Record(f"a{i}", 1, "s") for i in range(n_ai)]


def test_grouped_k_folds_keeps_groups_together_and_balances_classes():
    records = _records(200, 200)
    groups = list(range(400))
    for i in range(0, 60, 3):          # 20 groups of 3 near-duplicate human images
        groups[i + 1] = groups[i + 2] = groups[i]
    groups[200 + 5] = groups[200 + 6] = groups[200]  # one AI group of 3
    folds = grouped_k_folds(records, groups, k=5, seed=0)
    for g in set(groups):
        assert len({folds[i] for i, x in enumerate(groups) if x == g}) == 1
    for f in range(5):
        labels = [r.label for r, x in zip(records, folds) if x == f]
        assert abs(labels.count(0) - 40) <= 3 and abs(labels.count(1) - 40) <= 3
    assert grouped_k_folds(records, groups, k=5, seed=0) == folds  # deterministic


def test_grouped_holdout_fraction_and_groups():
    records = _records(100, 100)
    groups = list(range(200))
    groups[1] = groups[2] = 0
    rest, held = grouped_holdout(records, groups, 0.15, seed=1)
    assert len(rest) + len(held) == 200
    assert sum(r.label == 0 for r in held) == pytest.approx(15, abs=2)
    assert sum(r.label == 1 for r in held) == pytest.approx(15, abs=2)
    in_held = {r.path for r in held}
    assert len({p in in_held for p in ("h0", "h1", "h2")}) == 1


def test_near_duplicate_groups_follow_chains(tmp_path, art_image, monkeypatch):
    from attribution.data import eda

    # a~b (2 bits), b~c (2 bits) but a,c 4 bits apart; d far from all
    fake = {"a": 0b0000, "b": 0b0011, "c": 0b1111, "d": 2**63 - 1}
    monkeypatch.setattr(eda, "hash_records", lambda recs, workers=None: [fake[r.path] for r in recs])
    recs = [Record(p, 0, "s") for p in "abcd"]
    groups = near_duplicate_groups(recs, max_distance=2)
    assert groups[0] == groups[1] == groups[2] != groups[3]


def test_holm_adjust_matches_definition():
    p = [0.01, 0.04, None, 0.03, 0.005]
    adj = holm_adjust(p)
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    assert adj[4] == pytest.approx(0.02)
    assert adj[0] == pytest.approx(0.03)
    assert adj[3] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)
    assert adj[2] is None


def _load_cv():
    spec = importlib.util.spec_from_file_location("cross_validate", Path(__file__).resolve().parents[1] / "scripts" / "cross_validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aggregate_paired_tests(tmp_path):
    cv = _load_cv()
    rng = np.random.default_rng(0)
    fields = ["arch", "condition", "system", "n", "coverage", "accuracy", "precision", "recall", "fpr", "f1", "ece"]
    for f in range(5):
        rows = []
        for cond in ("clean", "jpeg_q30"):
            for system, f1 in [("provenance_only", 1.0 if cond == "clean" else None),
                               ("watermark_only", 0.99 + rng.normal(0, 0.002)),
                               ("cnn_only", 0.75 + rng.normal(0, 0.02)),
                               ("multi_signal", 1.0 if cond == "clean" else 0.97 + rng.normal(0, 0.005))]:
                rows.append({"arch": "resnet50", "condition": cond, "system": system, "n": 100, "coverage": 1,
                             "accuracy": f1, "precision": f1, "recall": f1 if f1 is not None else 0, "fpr": 0, "f1": f1, "ece": 0.01})
        path = tmp_path / f"fold_{f}" / "summary.csv"
        path.parent.mkdir()
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    report = cv.aggregate(tmp_path, k=5, alpha=0.05)
    assert report["folds_done"] == [0, 1, 2, 3, 4]
    with (tmp_path / "significance.csv").open() as fh:
        tests = {(r["condition"], r["comparison"]): r for r in csv.DictReader(fh)}
    cnn = tests[("jpeg_q30", "multi_signal vs cnn_only")]
    assert float(cnn["p_value"]) < 0.001 and cnn["significant_holm"] == "True"
    prov = tests[("jpeg_q30", "multi_signal vs provenance_only")]   # provenance F1 counted as 0
    assert float(prov["mean_f1_single"]) == 0.0 and prov["significant_p05"] == "True"
    same = tests[("clean", "multi_signal vs provenance_only")]      # 1.0 vs 1.0 in every fold
    assert same["p_value"] == "" and "undefined" in same["note"]
    with (tmp_path / "summary_mean_sd.csv").open() as fh:
        sd = {(r["condition"], r["system"]): r for r in csv.DictReader(fh)}
    assert float(sd[("clean", "multi_signal")]["f1_mean"]) == 1.0
    assert float(sd[("clean", "multi_signal")]["f1_sd"]) == 0.0
