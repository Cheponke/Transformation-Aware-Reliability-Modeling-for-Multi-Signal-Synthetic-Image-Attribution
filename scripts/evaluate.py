"""End-to-end evaluation of single-signal and hierarchical attribution.

Writes per-image signal outputs and summary metrics for every configured
transformation condition.

Experimental assumptions:
- A watermark is simulated at generation time on AI-labeled images only.
- When provenance is enabled, images are signed with a local thesis test CA
  using their ground-truth label. This is a controlled credential-adoption
  experiment, not evidence of real-world C2PA adoption.

Usage:
    python scripts/evaluate.py --arch resnet50
    python scripts/evaluate.py --arch resnet50 --max-images 100
    python scripts/evaluate.py --skip-provenance
"""
import argparse
import csv
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from attribution.config import load_config, splits_dir
from attribution.data.splits import read_split
from attribution.data.torch_dataset import eval_transform
from attribution.evaluation.metrics import (
    classification_report,
    expected_calibration_error,
    selective_report,
)
from attribution.fusion.decision import Evidence, decide
from attribution.models.cnn import build_model
from attribution.signals.forensic import calibrated_probs, predict_logits
from attribution.signals import provenance as prov
from attribution.signals import watermark
from attribution.transforms.social_media import build_conditions, delivery_format, encode


def _load_rgb(path: str) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")


def _binary_label(label: str) -> int:
    if label == "ai":
        return 1
    if label == "human":
        return 0
    raise ValueError(f"Cannot convert non-binary label {label!r} to a binary label")


def _ece_from_confidence(y_true, labels, confidence, n_bins: int = 15) -> float:
    """Top-label ECE; abstentions count as incorrect decisions."""
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray([_binary_label(x) if x in ("ai", "human") else -1 for x in labels])
    conf = np.clip(np.asarray(confidence, dtype=float), 0.0, 1.0)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf >= edges[i]) & (
            conf < edges[i + 1] if i < n_bins - 1 else conf <= edges[i + 1]
        )
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


class _SingleImageDataset(Dataset):
    def __init__(self, image: Image.Image, label: int, transform):
        self.image = image.copy()
        self.label = int(label)
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.transform(self.image), torch.tensor(float(self.label))


def _prediction_rows(
    records, condition_name, cfg, model, temperature, device,
    watermark_threshold, prov_signer, prov_ctx, use_provenance,
):
    """Run all signal channels on the same delivered pixel content."""
    rows = []
    jpeg_quality = int(cfg["preprocess"]["harmonize"]["jpeg_quality"])
    key = int(cfg["watermark"]["key"])
    alpha = float(cfg["watermark"]["alpha"])
    transform = build_conditions(cfg)[condition_name]
    fmt, quality = delivery_format(condition_name, cfg)
    tensor_transform = eval_transform(cfg)

    for record in records:
        original = _load_rgb(record.path)
        marked = watermark.embed(original, alpha, key) if record.label == 1 else original

        # Sign the generation-time image before simulated platform processing.
        if use_provenance:
            source_bytes = encode(marked, "JPEG", jpeg_quality)
            source_type = (
                cfg["provenance"]["ai_source_type"] if record.label == 1
                else cfg["provenance"]["human_source_type"]
            )
            signed_bytes, _ = prov.sign(
                source_bytes, "image/jpeg", source_type, prov_signer, prov_ctx
            )
            if transform is None:
                delivered_bytes = signed_bytes
                delivered_mime = "image/jpeg"
                with Image.open(io.BytesIO(signed_bytes)) as signed_image:
                    delivered = signed_image.convert("RGB")
            else:
                delivered = transform(marked)
                delivered_bytes = encode(delivered, fmt, int(quality))
                delivered_mime = "image/webp" if fmt == "WEBP" else "image/jpeg"
            provenance_result = prov.verify(delivered_bytes, delivered_mime, prov_ctx)
        else:
            delivered = marked if transform is None else transform(marked)
            delivered_bytes = encode(delivered, fmt, int(quality))
            delivered_mime = "image/webp" if fmt == "WEBP" else "image/jpeg"
            provenance_result = prov.ProvenanceResult(prov.ProvenanceStatus.ABSENT)

        wm_detected, wm_score = watermark.detect(delivered, key, watermark_threshold)
        loader = DataLoader(
            _SingleImageDataset(delivered, record.label, tensor_transform),
            batch_size=1,
            num_workers=0,
        )
        logits, _ = predict_logits(model, loader, device)
        prob_ai = float(calibrated_probs(logits, temperature)[0])

        fused = decide(
            Evidence(
                provenance=provenance_result,
                watermark_detected=bool(wm_detected),
                watermark_score=float(wm_score),
                cnn_prob_ai=prob_ai,
            ),
            abstain_low=float(cfg["fusion"]["abstain_low"]),
            abstain_high=float(cfg["fusion"]["abstain_high"]),
        )

        provenance_label = "uncertain"
        if (
            provenance_result.status is prov.ProvenanceStatus.VALID
            and provenance_result.claims_ai is not None
        ):
            provenance_label = "ai" if provenance_result.claims_ai else "human"
        # A negative watermark result means no detected mark, not proven human origin.
        watermark_label = "ai" if wm_detected else "uncertain"

        rows.append({
            "condition": condition_name,
            "path": record.path,
            "source": record.source,
            "y_true": int(record.label),
            "provenance_status": provenance_result.status.value,
            "provenance_claims_ai": provenance_result.claims_ai,
            "watermark_score": float(wm_score),
            "watermark_detected": bool(wm_detected),
            "cnn_prob_ai": prob_ai,
            "provenance_prediction": provenance_label,
            "watermark_prediction": watermark_label,
            "cnn_prediction": "ai" if prob_ai >= 0.5 else "human",
            "fusion_prediction": fused.label,
            "fusion_source": fused.source,
            "fusion_confidence": float(fused.confidence),
            "watermark_threshold": float(watermark_threshold),
        })
    return rows


def _summarize(rows, ece_bins):
    summary = []
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    systems = {
        "provenance_only": "provenance_prediction",
        "watermark_only": "watermark_prediction",
        "cnn_only": "cnn_prediction",
        "multi_signal": "fusion_prediction",
    }
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        y = [row["y_true"] for row in subset]
        for system, field in systems.items():
            predictions = [row[field] for row in subset]
            selective = selective_report(y, predictions)
            decided = [(yy, pp) for yy, pp in zip(y, predictions) if pp in ("ai", "human")]
            report = classification_report(
                [yy for yy, _ in decided],
                [_binary_label(pp) for _, pp in decided],
            ) if decided else {}
            item = {
                "condition": condition,
                "system": system,
                "n": len(subset),
                "coverage": float(selective.get("coverage", 0.0)),
                "accuracy": report.get("accuracy"),
                "precision": report.get("precision"),
                "recall": report.get("recall"),
                "f1": report.get("f1"),
                "fpr": report.get("fpr"),
                "ece": None,
                "brier_score": None,
                "mean_confidence": None,
                "abstention_rate": 1.0 - float(selective.get("coverage", 0.0)),
            }
            if system == "cnn_only":
                probs = [row["cnn_prob_ai"] for row in subset]
                item["ece"] = expected_calibration_error(y, probs, ece_bins)
                item["brier_score"] = float(np.mean((np.asarray(probs) - np.asarray(y)) ** 2))
            elif system == "multi_signal":
                item["ece"] = _ece_from_confidence(
                    y, predictions, [row["fusion_confidence"] for row in subset], ece_bins
                )
                item["mean_confidence"] = float(np.mean([row["fusion_confidence"] for row in subset]))
            summary.append(item)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--models", default="outputs/models")
    parser.add_argument("--out", default="outputs/evaluation")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-images", type=int, help="limit to this many images per class for a smoke test")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-provenance", action="store_true", help="run without C2PA signing/verification")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_dir = Path(args.models) / args.arch
    checkpoint = model_dir / "best.pt"
    calibration_file = model_dir / "calibration.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}. Train the model first.")
    if not calibration_file.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}. Calibrate the model first.")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available.")

    split_path = splits_dir(cfg) / f"{args.split}.csv"
    records = read_split(split_path)
    if args.max_images is not None:
        if args.max_images < 1:
            parser.error("--max-images must be at least 1")
        rng = random.Random(int(cfg["seed"]))
        selected = []
        for label in (0, 1):
            group = [record for record in records if record.label == label]
            selected.extend(rng.sample(group, min(args.max_images, len(group))))
        records = selected
    if not records or {record.label for record in records} != {0, 1}:
        raise RuntimeError(f"{split_path} must contain at least one human (0) and one AI (1) image.")

    model = build_model(args.arch, pretrained=False, dropout=cfg["model"]["dropout"])
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    temperature = float(json.loads(calibration_file.read_text())["temperature"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Invalid calibration temperature: {temperature}")

    threshold_path = Path("outputs/watermark/threshold.json")
    if threshold_path.is_file():
        watermark_threshold = float(json.loads(threshold_path.read_text())["threshold"])
        threshold_source = str(threshold_path)
    else:
        watermark_threshold = float(cfg["watermark"]["threshold"])
        threshold_source = "config fallback; run scripts/watermark_test.py to calibrate"
    if not np.isfinite(watermark_threshold):
        raise ValueError(f"Invalid watermark threshold: {watermark_threshold}")

    use_provenance = not args.skip_provenance
    prov_signer = prov_ctx = None
    if use_provenance:
        try:
            credentials = prov.make_credentials()
            prov_signer = prov.signer(credentials)
            prov_ctx = prov.context([credentials.root_pem])
        except ImportError as exc:
            raise RuntimeError(
                "C2PA dependencies are missing. Install with python -m pip install -e '.[provenance]' "
                "or use --skip-provenance for a partial run."
            ) from exc

    conditions = build_conditions(cfg)
    print(
        f"Evaluating {len(records)} images across {len(conditions)} conditions on {device}; "
        f"architecture={args.arch}; provenance={'on' if use_provenance else 'off'}; "
        f"watermark threshold={watermark_threshold:.5f} ({threshold_source})",
        flush=True,
    )

    all_rows = []
    for condition_name in conditions:
        print(f"== condition: {condition_name}", flush=True)
        rows = _prediction_rows(
            records, condition_name, cfg, model, temperature, device,
            watermark_threshold, prov_signer, prov_ctx, use_provenance,
        )
        all_rows.extend(rows)
        for item in _summarize(rows, int(cfg["evaluation"]["ece_bins"])):
            if item["system"] in ("cnn_only", "multi_signal"):
                print(
                    f"  {item['system']:>14}: acc={item['accuracy']} f1={item['f1']} "
                    f"coverage={item['coverage']:.3f} ece={item['ece']}",
                    flush=True,
                )

    summary = _summarize(all_rows, int(cfg["evaluation"]["ece_bins"]))
    clean_acc = {item["system"]: item["accuracy"] for item in summary if item["condition"] == "clean"}
    for item in summary:
        baseline = clean_acc.get(item["system"])
        item["delta_a"] = (
            float(baseline - item["accuracy"])
            if baseline is not None and item["accuracy"] is not None else None
        )

    out_dir = Path(args.out) / args.arch
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = out_dir / "per_image.csv"
    summary_path = out_dir / "summary.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "summary.json").write_text(json.dumps({
        "architecture": args.arch,
        "split": args.split,
        "n_images": len(records),
        "conditions": list(conditions),
        "temperature": temperature,
        "watermark_threshold": watermark_threshold,
        "watermark_threshold_source": threshold_source,
        "provenance_enabled": use_provenance,
        "notes": [
            "Watermarks are simulated on AI-labeled images only.",
            "When enabled, provenance claims are generated from ground-truth labels using a local test CA.",
            "These controlled conditions do not represent universal real-world adoption of watermarks or C2PA.",
            "Provenance-only and watermark-only systems abstain when their signal cannot support a decision.",
            "Classification metrics are computed on decided images; coverage reports the share receiving a decision.",
        ],
        "results": summary,
    }, indent=2))
    print(f"Per-image results: {per_image_path}")
    print(f"Summary results:   {summary_path}")
    print(f"JSON report:       {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
