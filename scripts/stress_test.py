"""Compression stress test (manuscript 3.5.2, 3.7.2): test accuracy under each social-media transform.

Evaluates a trained checkpoint on the test split once per condition in
transforms/social_media.py (clean, JPEG 90/70/50/30, WebP, downscaling,
recompression) and reports accuracy, precision, recall, F1, FPR, ECE and
delta-A = A_clean - A_condition.

Usage: python scripts/stress_test.py --arch resnet18
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from attribution.config import load_config, splits_dir
from attribution.data.splits import read_split
from attribution.data.torch_dataset import ImageRecords, eval_transform
from attribution.evaluation.metrics import classification_report, delta_accuracy, expected_calibration_error
from attribution.models.cnn import build_model
from attribution.signals.forensic import calibrated_probs, predict_logits
from attribution.transforms.social_media import build_conditions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--models", default="outputs/models")
    ap.add_argument("--out", default="outputs/stress_test")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = Path(args.models) / args.arch
    model = build_model(args.arch, pretrained=False, dropout=cfg["model"]["dropout"])
    model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device))
    temperature = json.loads((model_dir / "calibration.json").read_text())["temperature"]
    records = read_split(splits_dir(cfg) / f"{args.split}.csv")

    rows = []
    for name, degradation in build_conditions(cfg).items():
        ds = ImageRecords(records, eval_transform(cfg), degradation)
        loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], num_workers=cfg["train"]["num_workers"])
        logits, labels = predict_logits(model, loader, device)
        probs = calibrated_probs(logits, temperature)
        report = classification_report(labels, [int(p >= 0.5) for p in probs])
        row = {"condition": name, **{k: report[k] for k in ("accuracy", "precision", "recall", "f1", "fpr")}}
        row["ece"] = expected_calibration_error(labels, probs, cfg["evaluation"]["ece_bins"])
        rows.append(row)
        print(f"{name:>14}: acc={row['accuracy']:.4f} f1={row['f1']:.4f} fpr={row['fpr']:.4f} ece={row['ece']:.4f}", flush=True)

    clean = rows[0]["accuracy"]
    for row in rows:
        row["delta_a"] = delta_accuracy(clean, row["accuracy"])

    out = Path(args.out) / args.arch
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "results.json").write_text(json.dumps({"arch": args.arch, "temperature": temperature, "rows": rows}, indent=1))
    print(f"results: {out / 'results.csv'}")


if __name__ == "__main__":
    main()
