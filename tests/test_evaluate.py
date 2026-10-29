"""Metric aggregation in scripts/evaluate.py (the fusion evaluation)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from attribution.evaluation.metrics import expected_calibration_error  # noqa: E402
from attribution.fusion.decision import Evidence, decide  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "evaluate_script", Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
)
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)

BAND = dict(abstain_low=0.35, abstain_high=0.65)


def _row(y, prov="uncertain", wm="uncertain", p_ai=0.5, condition="clean", wm_conf=0.98):
    fused = decide(
        Evidence(
            provenance=None if prov == "uncertain" else evaluate.prov.ProvenanceResult(
                evaluate.prov.ProvenanceStatus.VALID, claims_ai=prov == "ai"),
            watermark_detected=wm == "ai",
            cnn_prob_ai=p_ai,
        ),
        watermark_confidence=wm_conf,
        **BAND,
    )
    return {
        "condition": condition,
        "y_true": y,
        "provenance_prediction": prov,
        "watermark_prediction": wm,
        "cnn_prediction": "ai" if p_ai >= 0.5 else "human",
        "fusion_prediction": fused.label,
        "fusion_source": fused.source,
        "cnn_prob_ai": p_ai,
        "provenance_confidence": 0.99 if prov != "uncertain" else None,
        "watermark_confidence": wm_conf if wm == "ai" else None,
        "cnn_confidence": max(p_ai, 1 - p_ai),
        "fusion_confidence": fused.confidence,
    }


def _by_system(summary, condition="clean"):
    return {s["system"]: s for s in summary if s["condition"] == condition}


def test_summary_has_all_four_systems_per_condition():
    rows = [_row(1, p_ai=0.9), _row(0, p_ai=0.1), _row(1, p_ai=0.9, condition="jpeg_q30")]
    summary = evaluate.summarize(rows, ece_bins=15)
    assert {(s["condition"], s["system"]) for s in summary} == {
        (c, s) for c in ("clean", "jpeg_q30") for s in ("provenance_only", "watermark_only", "cnn_only", "multi_signal")
    }


def test_single_channels_abstain_and_fusion_falls_back():
    rows = [
        _row(1, prov="ai", wm="ai", p_ai=0.2),   # provenance decides (correctly), CNN alone is wrong
        _row(0, prov="human", p_ai=0.8),         # provenance decides (correctly), CNN alone is wrong
        _row(1, wm="ai", p_ai=0.4),              # watermark decides
        _row(0, p_ai=0.1),                       # CNN decides
        _row(1, p_ai=0.5),                       # CNN abstains -> fusion uncertain
    ]
    s = _by_system(evaluate.summarize(rows, ece_bins=15))
    assert s["provenance_only"]["coverage"] == pytest.approx(2 / 5)
    assert s["provenance_only"]["accuracy"] == 1.0
    assert s["watermark_only"]["n_decided"] == 2
    assert s["cnn_only"]["coverage"] == 1.0
    assert s["cnn_only"]["accuracy"] == pytest.approx(2 / 5)  # 0.5 counts as AI: rows 4 and 5 right
    fusion = s["multi_signal"]
    assert fusion["n_decided"] == 4 and fusion["accuracy"] == 1.0
    assert fusion["decided_by_provenance"] == pytest.approx(2 / 5)
    assert fusion["decided_by_watermark"] == pytest.approx(1 / 5)
    assert fusion["decided_by_cnn"] == pytest.approx(1 / 5)


def test_fusion_uses_measured_watermark_confidence():
    rows = [_row(1, wm="ai", p_ai=0.5, wm_conf=0.987)]
    assert rows[0]["fusion_source"] == "watermark"
    assert rows[0]["fusion_confidence"] == pytest.approx(0.987)
    s = _by_system(evaluate.summarize(rows, ece_bins=15))
    assert s["multi_signal"]["mean_confidence"] == pytest.approx(0.987)
    assert s["multi_signal"]["ece"] == pytest.approx(1 - 0.987)


def test_top_label_ece_matches_cnn_ece():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.5 * y + 0.25 + rng.normal(0, 0.2, 500), 0.001, 0.999)
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p)
    assert evaluate.top_label_ece(pred == y, conf, 15) == pytest.approx(expected_calibration_error(y, p, 15))


def test_delta_a_inputs_present_when_nothing_decided():
    rows = [_row(1, p_ai=0.9), _row(0, p_ai=0.1)]
    s = _by_system(evaluate.summarize(rows, ece_bins=15))
    assert s["provenance_only"]["n_decided"] == 0
    assert s["provenance_only"]["accuracy"] is None and s["provenance_only"]["ece"] is None


def test_one_sided_watermark_rates_use_all_images():
    # 3 AI (2 detected), 4 human (1 false alarm); the watermark never says "human"
    rows = [_row(1, wm="ai"), _row(1, wm="ai"), _row(1), _row(0, wm="ai"), _row(0), _row(0), _row(0)]
    s = _by_system(evaluate.summarize(rows, ece_bins=15))["watermark_only"]
    assert s["recall"] == pytest.approx(2 / 3)
    assert s["fpr"] == pytest.approx(1 / 4)  # not 1.0, which a decided-only FPR would give
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["f1"] == pytest.approx(2 / 3)


def test_rates_when_channel_never_decides():
    s = _by_system(evaluate.summarize([_row(1, p_ai=0.9), _row(0, p_ai=0.1)], ece_bins=15))["provenance_only"]
    assert s["recall"] == 0.0 and s["fpr"] == 0.0 and s["coverage"] == 0.0


def test_resummarize_matches_direct_summary(tmp_path):
    import csv
    import json

    rows = [_row(1, prov="ai", p_ai=0.2), _row(0, p_ai=0.8), _row(1, wm="ai", p_ai=0.4),
            _row(0, p_ai=0.1), _row(1, p_ai=0.5), _row(1, p_ai=0.9, condition="jpeg_q30")]
    with (tmp_path / "per_image.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (tmp_path / "summary.json").write_text(json.dumps({"architecture": "x", "notes": ["keep me"]}))
    evaluate.resummarize(tmp_path, ece_bins=15)
    meta = json.loads((tmp_path / "summary.json").read_text())
    assert meta["architecture"] == "x" and "keep me" in meta["notes"]
    direct = evaluate.final_summary(rows, ece_bins=15)
    for a, b in zip(meta["results"], direct):
        assert a.keys() == b.keys()
        for k in a:
            assert a[k] == pytest.approx(b[k]) if isinstance(b[k], float) else a[k] == b[k]


def test_exclusion_sensitivity_drops_flagged_test_images(tmp_path):
    import csv
    import json

    rows = []
    for condition in ("clean", "jpeg_q30"):
        for i, (y, p_ai) in enumerate([(1, 0.9), (1, 0.2), (0, 0.1), (0, 0.8), (1, 0.95), (0, 0.05)]):
            r = _row(y, p_ai=p_ai, condition=condition)
            r["path"] = f"data/harmonized/test/{i:06d}.jpg"
            rows.append(r)
    with (tmp_path / "per_image.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (tmp_path / "summary.json").write_text(json.dumps({"split": "test"}))
    # test images 1 and 3 (the CNN's two mistakes) are flagged; the val image is ignored
    pairs = tmp_path / "near_duplicates.csv"
    with pairs.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split_a", "path_a", "label_a", "source_a", "split_b", "path_b", "label_b", "source_b", "hash_distance"])
        w.writerow(["train", "data/harmonized/train/000009.jpg", 1, "s", "test", "data/harmonized/test/000001.jpg", 1, "s", 0])
        w.writerow(["val", "data/harmonized/val/000002.jpg", 0, "s", "test", "./data/harmonized/test/000003.jpg", 0, "s", 3])

    comparison = evaluate.exclusion_sensitivity(tmp_path, pairs, ece_bins=15)
    cnn = {r["condition"]: r for r in comparison if r["system"] == "cnn_only"}
    assert cnn["clean"]["n_all"] == 6 and cnn["clean"]["n_excluded"] == 2
    assert cnn["clean"]["accuracy_all"] == pytest.approx(4 / 6)
    assert cnn["clean"]["accuracy_without"] == pytest.approx(1.0)
    assert cnn["clean"]["accuracy_change"] == pytest.approx(1.0 - 4 / 6)
    info = json.loads((tmp_path / "summary_excl_near_duplicates.json").read_text())
    assert info["excluded_images"] == 2 and info["flagged_images_in_split"] == 2
    assert (tmp_path / "summary_excl_near_duplicates.csv").is_file()


def test_exclusion_sensitivity_rejects_mismatched_splits(tmp_path):
    import csv

    r = _row(1, p_ai=0.9)
    r["path"] = "data/harmonized/test/000000.jpg"
    with (tmp_path / "per_image.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r))
        w.writeheader()
        w.writerow(r)
    pairs = tmp_path / "near_duplicates.csv"
    pairs.write_text("split_a,path_a,label_a,source_a,split_b,path_b,label_b,source_b,hash_distance\n"
                     "train,x/train/1.jpg,1,s,test,other/test/9.jpg,1,s,0\n")
    with pytest.raises(RuntimeError, match="same splits"):
        evaluate.exclusion_sensitivity(tmp_path, pairs, ece_bins=15)
