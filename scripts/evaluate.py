"""Multi-signal fusion evaluation (manuscript sections 3.2, 3.7; research questions 2-3).

For every test image and every social-media condition, the three channels are
run on the same image and combined by the hierarchical fusion rule
(valid provenance -> detected watermark -> calibrated CNN):

1. AI images get the generation-time watermark (I_w = I + alpha * W) at the
   configured alpha; human images stay unmarked.
2. The image is saved as JPEG at the harmonized quality. With provenance on,
   that file is signed with a test CA (AI: trainedAlgorithmicMedia, human:
   digitalCreation).
3. Clean condition: the platform serves that file unchanged, and every channel
   reads it. Other conditions: the transform is applied to the image (exactly
   as in the CNN and watermark stress tests), the watermark detector and the
   CNN read the transformed pixels, and the C2PA check reads the transformed
   image saved in the platform's delivery format. Re-encoding drops the
   manifest, as most platforms do (Pillow writes no C2PA data).

Before the test split is touched, the watermark detector's precision
P(AI | detected) is measured on the validation split, pooled over every
condition because the detector cannot tell how an image was processed. That
value is the confidence fusion gives to watermark-based decisions.

Single-channel systems abstain when their signal cannot support a decision
(no valid credential, no watermark detected); the CNN alone always decides.
Accuracy, F1 and ECE are computed on the images that received a decision, and
coverage reports the share that did.

Writes outputs/evaluation/<arch>/{per_image.csv, summary.csv, summary.json,
watermark_confidence.json}.
Usage: python scripts/evaluate.py --arch resnet50 [--max-images 50] [--skip-provenance]
"""
import argparse
import csv
import io
import json
import random
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from attribution.config import load_config, splits_dir
from attribution.data.splits import read_split
from attribution.data.torch_dataset import eval_transform
from attribution.evaluation.metrics import classification_report, expected_calibration_error
from attribution.fusion.decision import PROVENANCE_CONFIDENCE, Evidence, decide
from attribution.models.cnn import build_model
from attribution.signals import provenance as prov
from attribution.signals import watermark
from attribution.signals.forensic import calibrated_probs, predict_logits
from attribution.transforms.social_media import build_conditions, delivery_format, encode

MIME = {"JPEG": "image/jpeg", "WEBP": "image/webp"}

# system name -> (prediction column, confidence column) in per_image.csv
SYSTEMS = {
    "provenance_only": ("provenance_prediction", "provenance_confidence"),
    "watermark_only": ("watermark_prediction", "watermark_confidence"),
    "cnn_only": ("cnn_prediction", "cnn_confidence"),
    "multi_signal": ("fusion_prediction", "fusion_confidence"),
}


def _load_rgb(path: str) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")


def _decode(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as f:
        return f.convert("RGB")


def _binary_label(label: str) -> int:
    if label == "ai":
        return 1
    if label == "human":
        return 0
    raise ValueError(f"Cannot convert non-binary label {label!r} to a binary label")


def top_label_ece(correct, confidence, n_bins: int = 15) -> float:
    """ECE of decided predictions: confidence of the predicted label vs. how often it is right.

    Bins span [0.5, 1] exactly as `expected_calibration_error`, so for the CNN alone
    (which decides every image) both functions give the same value.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    c = np.asarray(correct, dtype=float)
    conf = np.asarray(confidence, dtype=float)
    if c.shape != conf.shape:
        raise ValueError("correct and confidence must have equal lengths")
    if c.size == 0:
        return 0.0
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(c[mask].mean() - conf[mask].mean())
    return float(ece)


def estimate_watermark_confidence(records, cfg, threshold: float, alpha: float, key: int) -> dict:
    """P(AI | watermark detected) on labelled images, pooled over every condition.

    AI images are marked at `alpha`, human images are not, exactly as in the
    test run. Rates use a +0.5/+1 (Jeffreys) correction so a zero false-alarm
    count cannot produce a confidence of exactly 1. Precision assumes equal
    class priors, matching the balanced test split.
    """
    conditions = build_conditions(cfg)
    tp = fp = n_marked = n_unmarked = 0
    for record in records:
        img = _load_rgb(record.path)
        is_ai = record.label == 1
        marked = watermark.embed(img, alpha, key) if is_ai else img
        for transform in conditions.values():
            detected, _ = watermark.detect(marked if transform is None else transform(marked), key, threshold)
            if is_ai:
                n_marked += 1
                tp += int(detected)
            else:
                n_unmarked += 1
                fp += int(detected)
    if n_marked == 0 or n_unmarked == 0:
        raise RuntimeError("Watermark confidence needs both AI and human validation images.")
    tpr = (tp + 0.5) / (n_marked + 1)
    fpr = (fp + 0.5) / (n_unmarked + 1)
    return {
        "confidence": tpr / (tpr + fpr),
        "tpr": tp / n_marked,
        "fpr": fp / n_unmarked,
        "tp": tp,
        "fp": fp,
        "n_marked_scores": n_marked,
        "n_unmarked_scores": n_unmarked,
        "conditions": list(conditions),
        "alpha": alpha,
        "threshold": threshold,
        "prior": "equal class priors (balanced test split)",
    }


class _ImageList(Dataset):
    """In-memory images for one CNN batch."""

    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.transform(self.images[index]), torch.tensor(float(self.labels[index]))


def _deliver(record, transform, condition_name, cfg, prov_signer, prov_ctx, use_provenance):
    """The image the channels see and the provenance result for one record under one condition."""
    alpha = float(cfg["watermark"]["alpha"])
    key = int(cfg["watermark"]["key"])
    original = _load_rgb(record.path)
    # Simulate a generation-time watermark on AI images only.
    marked = watermark.embed(original, alpha, key) if record.label == 1 else original

    # The file as published (before any platform processing).
    source_bytes = encode(marked, "JPEG", int(cfg["preprocess"]["harmonize"]["jpeg_quality"]))
    if use_provenance:
        p = cfg["provenance"]
        source_type = p["ai_source_type"] if record.label == 1 else p["human_source_type"]
        source_bytes, _ = prov.sign(source_bytes, "image/jpeg", source_type, prov_signer, prov_ctx)

    if transform is None:
        delivered = _decode(source_bytes)
        delivered_bytes, delivered_mime = source_bytes, "image/jpeg"
    else:
        delivered = transform(marked)
        fmt, quality = delivery_format(condition_name, cfg)
        delivered_bytes, delivered_mime = encode(delivered, fmt, int(quality)), MIME[fmt]

    if use_provenance:
        provenance_result = prov.verify(delivered_bytes, delivered_mime, prov_ctx)
    else:
        provenance_result = prov.ProvenanceResult(prov.ProvenanceStatus.ABSENT)
    return delivered, provenance_result


def _prediction_rows(records, condition_name, cfg, model, temperature, device, watermark_threshold,
                     watermark_confidence, prov_signer, prov_ctx, use_provenance, batch_size=64):
    """Run all channels and the fusion rule on every record under one condition."""
    rows = []
    key = int(cfg["watermark"]["key"])
    transform = build_conditions(cfg)[condition_name]
    tensor_transform = eval_transform(cfg)
    abstain_low = float(cfg["fusion"]["abstain_low"])
    abstain_high = float(cfg["fusion"]["abstain_high"])

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        delivered = [_deliver(r, transform, condition_name, cfg, prov_signer, prov_ctx, use_provenance) for r in batch]
        images = [img for img, _ in delivered]

        loader = DataLoader(_ImageList(images, [r.label for r in batch], tensor_transform),
                            batch_size=batch_size, num_workers=0)
        logits, _ = predict_logits(model, loader, device)
        probs = calibrated_probs(logits, temperature)

        for record, (image, provenance_result), prob_ai in zip(batch, delivered, probs):
            prob_ai = float(prob_ai)
            wm_detected, wm_score = watermark.detect(image, key, watermark_threshold)

            # Hierarchical fusion: provenance -> watermark -> CNN.
            fused = decide(
                Evidence(provenance=provenance_result, watermark_detected=bool(wm_detected),
                         watermark_score=float(wm_score), cnn_prob_ai=prob_ai),
                abstain_low=abstain_low,
                abstain_high=abstain_high,
                watermark_confidence=watermark_confidence,
            )

            # A missing or invalid credential is not a human prediction.
            provenance_label = "uncertain"
            if provenance_result.status is prov.ProvenanceStatus.VALID and provenance_result.claims_ai is not None:
                provenance_label = "ai" if provenance_result.claims_ai else "human"
            # Failure to detect a watermark is not proof of human origin.
            watermark_label = "ai" if wm_detected else "uncertain"
            cnn_label = "ai" if prob_ai >= 0.5 else "human"

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
                "cnn_prediction": cnn_label,
                "fusion_prediction": fused.label,
                "fusion_source": fused.source,
                "provenance_confidence": PROVENANCE_CONFIDENCE if provenance_label != "uncertain" else None,
                "watermark_confidence": float(watermark_confidence) if wm_detected else None,
                "cnn_confidence": max(prob_ai, 1.0 - prob_ai),
                "fusion_confidence": float(fused.confidence),
                "watermark_threshold": float(watermark_threshold),
            })
    return rows


def summarize(rows, ece_bins: int) -> list[dict]:
    """Metrics per condition and system; classification metrics and ECE use decided images only."""
    summary = []
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        n = len(subset)
        for system, (pred_field, conf_field) in SYSTEMS.items():
            decided = [row for row in subset if row[pred_field] in ("ai", "human")]
            y = [int(row["y_true"]) for row in decided]
            pred = [_binary_label(row[pred_field]) for row in decided]
            conf = [float(row[conf_field]) for row in decided]
            report = classification_report(y, pred) if decided else {}
            accuracy = report.get("accuracy")
            mean_conf = float(np.mean(conf)) if conf else None
            item = {
                "condition": condition,
                "system": system,
                "n": n,
                "n_decided": len(decided),
                "n_abstained": n - len(decided),
                "coverage": len(decided) / n if n else 0.0,
                "accuracy": accuracy,
                "precision": report.get("precision"),
                "recall": report.get("recall"),
                "f1": report.get("f1"),
                "fpr": report.get("fpr"),
                "tp": report.get("tp"),
                "tn": report.get("tn"),
                "fp": report.get("fp"),
                "fn": report.get("fn"),
                "ece": top_label_ece([p == t for p, t in zip(pred, y)], conf, ece_bins) if decided else None,
                "brier_score": None,
                "mean_confidence": mean_conf,
                "confidence_accuracy_gap": mean_conf - accuracy if mean_conf is not None and accuracy is not None else None,
                "decided_by_provenance": None,
                "decided_by_watermark": None,
                "decided_by_cnn": None,
            }
            if system == "cnn_only":
                probs = np.asarray([float(row["cnn_prob_ai"]) for row in subset])
                truth = np.asarray([int(row["y_true"]) for row in subset], dtype=float)
                item["brier_score"] = float(np.mean((probs - truth) ** 2))
                # Same definition used in calibration and the stress test (CNN never abstains).
                item["ece"] = expected_calibration_error(truth, probs, ece_bins)
            if system == "multi_signal":
                for source in ("provenance", "watermark", "cnn"):
                    item[f"decided_by_{source}"] = (
                        sum(row["fusion_source"] == source and row["fusion_prediction"] != "uncertain" for row in subset) / n
                        if n else 0.0
                    )
            summary.append(item)
    return summary


def _c2pa_version() -> str | None:
    try:
        return metadata.version("c2pa-python")
    except metadata.PackageNotFoundError:
        return None


def _fmt(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--models", default="outputs/models")
    parser.add_argument("--watermark-dir", default="outputs/watermark",
                        help="where scripts/watermark_test.py wrote threshold.json")
    parser.add_argument("--out", default="outputs/evaluation")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-images", type=int, help="limit to this many images per class (test and validation) for a smoke test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-provenance", action="store_true", help="run without C2PA signing and verification")
    parser.add_argument("--watermark-confidence", type=float,
                        help="use this P(AI | watermark detected) instead of measuring it on the validation split")
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
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    rng = random.Random(int(cfg["seed"]))

    def per_class_subset(records):
        if args.max_images is None:
            return records
        selected = []
        for label in (0, 1):
            group = [r for r in records if r.label == label]
            selected.extend(rng.sample(group, min(args.max_images, len(group))))
        return selected

    split_path = splits_dir(cfg) / f"{args.split}.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    records = per_class_subset(read_split(split_path))
    if not records or {r.label for r in records} != {0, 1}:
        raise RuntimeError(f"{split_path} must contain at least one human (0) and one AI (1) image.")

    model = build_model(args.arch, pretrained=False, dropout=cfg["model"]["dropout"])
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()

    temperature = float(json.loads(calibration_file.read_text())["temperature"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Invalid calibration temperature: {temperature}")

    threshold_path = Path(args.watermark_dir) / "threshold.json"
    if threshold_path.is_file():
        watermark_threshold = float(json.loads(threshold_path.read_text())["threshold"])
        threshold_source = str(threshold_path)
    else:
        watermark_threshold = float(cfg["watermark"]["threshold"])
        threshold_source = "config fallback; run scripts/watermark_test.py to calibrate"
        print(f"WARNING: {threshold_path} not found, using the config fallback threshold {watermark_threshold}", flush=True)
    if not np.isfinite(watermark_threshold):
        raise ValueError(f"Invalid watermark threshold: {watermark_threshold}")

    alpha = float(cfg["watermark"]["alpha"])
    key = int(cfg["watermark"]["key"])
    out_dir = Path(args.out) / args.arch
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.watermark_confidence is not None:
        if not 0.5 <= args.watermark_confidence <= 1.0:
            parser.error("--watermark-confidence must be between 0.5 and 1")
        wm_conf = {"confidence": args.watermark_confidence, "source": "command line"}
    else:
        val_path = splits_dir(cfg) / "val.csv"
        if not val_path.is_file():
            raise FileNotFoundError(f"Validation split not found: {val_path} (needed to measure watermark confidence)")
        val_records = per_class_subset(read_split(val_path))
        print(f"Measuring watermark confidence on {len(val_records)} validation images (alpha={alpha})", flush=True)
        wm_conf = estimate_watermark_confidence(val_records, cfg, watermark_threshold, alpha, key)
        wm_conf["source"] = str(val_path)
        print(f"  P(AI | detected) = {wm_conf['confidence']:.4f}  (TPR {wm_conf['tpr']:.4f}, FPR {wm_conf['fpr']:.4f})", flush=True)
    (out_dir / "watermark_confidence.json").write_text(json.dumps(wm_conf, indent=1))
    watermark_confidence = float(wm_conf["confidence"])

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
    if not conditions:
        raise RuntimeError("No evaluation conditions were configured.")
    print(
        f"Evaluating {len(records)} images across {len(conditions)} conditions on {device}; "
        f"architecture={args.arch}; provenance={'on' if use_provenance else 'off'}; "
        f"watermark alpha={alpha}, threshold={watermark_threshold:.5f} ({threshold_source})",
        flush=True,
    )

    ece_bins = int(cfg["evaluation"]["ece_bins"])
    all_rows = []
    for condition_name in conditions:
        print(f"== condition: {condition_name}", flush=True)
        rows = _prediction_rows(records, condition_name, cfg, model, temperature, device, watermark_threshold,
                                watermark_confidence, prov_signer, prov_ctx, use_provenance, args.batch_size)
        all_rows.extend(rows)
        for item in summarize(rows, ece_bins):
            print(f"  {item['system']:>15}: coverage={item['coverage']:.3f} acc={_fmt(item['accuracy'])} "
                  f"f1={_fmt(item['f1'])} fpr={_fmt(item['fpr'])} ece={_fmt(item['ece'])}", flush=True)

    summary = summarize(all_rows, ece_bins)

    # Delta-A is relative to clean accuracy for the same system.
    clean_acc = {item["system"]: item["accuracy"] for item in summary if item["condition"] == "clean"}
    for item in summary:
        baseline = clean_acc.get(item["system"])
        item["delta_a"] = (
            float(baseline - item["accuracy"]) if baseline is not None and item["accuracy"] is not None else None
        )

    per_image_path = out_dir / "per_image.csv"
    summary_path = out_dir / "summary.csv"
    for path, data in [(per_image_path, all_rows), (summary_path, summary)]:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    (out_dir / "summary.json").write_text(json.dumps({
        "architecture": args.arch,
        "split": args.split,
        "n_images": len(records),
        "conditions": list(conditions),
        "temperature": temperature,
        "watermark_alpha": alpha,
        "watermark_threshold": watermark_threshold,
        "watermark_threshold_source": threshold_source,
        "watermark_confidence": wm_conf,
        "provenance_enabled": use_provenance,
        "provenance_confidence": PROVENANCE_CONFIDENCE,
        "c2pa_python_version": _c2pa_version() if use_provenance else None,
        "abstain_band": [float(cfg["fusion"]["abstain_low"]), float(cfg["fusion"]["abstain_high"])],
        "notes": [
            "Watermarks are simulated on AI-labeled images only.",
            "When enabled, provenance claims are generated from ground-truth labels using a local test CA.",
            "These controlled conditions do not represent universal real-world adoption of watermarks or C2PA.",
            "Re-encoded files carry no manifest (stripping platform); soft binding is not used.",
            "Provenance-only and watermark-only systems abstain when their signal cannot support a decision.",
            "Classification metrics and ECE are computed on decided images; coverage reports the share receiving a decision.",
        ],
        "results": summary,
    }, indent=2))

    print(f"Per-image results: {per_image_path}")
    print(f"Summary results: {summary_path}")
    print(f"JSON report: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
