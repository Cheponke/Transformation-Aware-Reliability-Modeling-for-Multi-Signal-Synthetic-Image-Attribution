"""Watermark-adoption sweep: recombining the watermarked and unmarked runs."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("adoption_sweep", SCRIPTS / "adoption_sweep.py")
sweep_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep_mod)
import evaluate as ev  # noqa: E402

from attribution.fusion.decision import Evidence, decide  # noqa: E402
from attribution.signals.provenance import ProvenanceResult, ProvenanceStatus  # noqa: E402

BAND = (0.35, 0.65)
THRESHOLD, CONF = 0.036, 0.998


def _row(condition, path, y, prob, score, marked_channel=True):
    detected = marked_channel and score >= THRESHOLD
    prov = ProvenanceResult(ProvenanceStatus.VALID if condition == "clean" else ProvenanceStatus.ABSENT,
                            bool(y) if condition == "clean" else None)
    fused = decide(Evidence(provenance=prov, watermark_detected=detected, cnn_prob_ai=prob),
                   abstain_low=BAND[0], abstain_high=BAND[1], watermark_confidence=CONF)
    plabel = ("ai" if y else "human") if condition == "clean" else "uncertain"
    return {"condition": condition, "path": path, "source": "s", "y_true": y,
            "provenance_status": prov.status.value, "provenance_claims_ai": prov.claims_ai,
            "watermark_score": score, "watermark_detected": detected, "cnn_prob_ai": prob,
            "provenance_prediction": plabel, "watermark_prediction": "ai" if detected else "uncertain",
            "cnn_prediction": "ai" if prob >= 0.5 else "human", "fusion_prediction": fused.label,
            "fusion_source": fused.source, "provenance_confidence": 0.99 if plabel != "uncertain" else None,
            "watermark_confidence": CONF if detected else None, "cnn_confidence": max(prob, 1 - prob),
            "fusion_confidence": fused.confidence, "watermark_threshold": THRESHOLD}


@pytest.fixture
def runs(tmp_path):
    rng = np.random.default_rng(0)
    marked, unmarked = [], []
    for condition in ("clean", "jpeg_q30"):
        for i in range(200):
            y = i % 2
            p_m, p_u = (rng.uniform(0.2, 0.9), rng.uniform(0.4, 1.0)) if y else (rng.uniform(0, 0.6),) * 2
            s_u = rng.normal(0, 0.01)
            s_m = 0.2 + rng.normal(0, 0.01) if y else s_u
            marked.append(_row(condition, f"t/{i}.jpg", y, p_m, s_m))
            unmarked.append(_row(condition, f"t/{i}.jpg", y, p_u, s_u, marked_channel=False))
    for name, rows in [("m", marked), ("m_no_watermark", unmarked)]:
        d = tmp_path / name
        d.mkdir()
        with (d / "per_image.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (tmp_path / "m" / "summary.json").write_text(json.dumps({"watermark_confidence": {"confidence": CONF}}))
    return tmp_path, marked


def _by(rows, condition, system):
    return next(r for r in rows if r["condition"] == condition and r["system"] == system)


def test_full_adoption_reproduces_the_watermarked_run(runs):
    eval_dir, marked = runs
    pairs, conf = sweep_mod.load_pairs(eval_dir, "m")
    out = sweep_mod.sweep(pairs, conf, BAND, [1.0], [1.0], repeats=3, seed=0, ece_bins=15)
    direct = ev.summarize(marked, 15)
    for row in out:
        ref = _by(direct, row["condition"], row["system"])
        for m in sweep_mod.METRICS:
            assert row[m] == pytest.approx(ref[m]) if ref[m] is not None else row[m] is None


def test_zero_adoption_turns_the_watermark_off_for_ai_images(runs):
    eval_dir, _ = runs
    pairs, conf = sweep_mod.load_pairs(eval_dir, "m")
    out = sweep_mod.sweep(pairs, conf, BAND, [0.0, 0.5], [1.0], repeats=4, seed=0, ece_bins=15)
    wm0 = _by([r for r in out if r["watermark_adoption"] == 0.0], "jpeg_q30", "watermark_only")
    assert wm0["recall"] == 0.0
    half = [r for r in out if r["watermark_adoption"] == 0.5 and r["condition"] == "jpeg_q30" and r["system"] == "watermark_only"]
    assert len(half) == 4 and all(0.35 < r["recall"] < 0.65 for r in half)
    summary, crossover = sweep_mod.summarize_draws(out)
    assert {c["condition"] for c in crossover} == {"clean", "jpeg_q30"}


def test_no_provenance_removes_clean_shortcut(runs):
    eval_dir, _ = runs
    pairs, conf = sweep_mod.load_pairs(eval_dir, "m")
    out = sweep_mod.sweep(pairs, conf, BAND, [1.0], [0.0], repeats=2, seed=0, ece_bins=15)
    assert _by(out, "clean", "provenance_only")["coverage"] == 0.0
    assert _by(out, "clean", "multi_signal")["coverage"] < 1.0  # human images now fall to the CNN, which can abstain


def test_mismatched_runs_raise(runs):
    eval_dir, _ = runs
    path = eval_dir / "m_no_watermark" / "per_image.csv"
    rows = list(csv.DictReader(path.open()))[1:]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(RuntimeError, match="same test split"):
        sweep_mod.load_pairs(eval_dir, "m")
