"""Run the EDA on the prepared splits and write a report to outputs/eda/.

Usage: python scripts/eda.py [--max-per-split 2000]
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from attribution.config import load_config, splits_dir
from attribution.data.eda import channel_mean_std, class_balance, cross_split_duplicates, image_stats
from attribution.data.splits import read_split


def summarize(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-per-split", type=int, default=2000, help="images sampled per split for pixel stats")
    ap.add_argument("--raw", action="store_true", help="analyze the original files instead of the harmonized ones")
    ap.add_argument("--out", default=None, help="default: outputs/eda (outputs/eda_raw with --raw)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    processed = Path(cfg["data"]["processed_dir"]) if args.raw else splits_dir(cfg)
    size = cfg["preprocess"]["image_size"]
    rng = random.Random(cfg["seed"])

    splits = {s: read_split(processed / f"{s}.csv") for s in cfg["data"]["split"]}
    report = {"counts": {s: len(rs) for s, rs in splits.items()}}
    report["class_balance"] = {s: class_balance(rs) for s, rs in splits.items()}

    per_image = []
    for split, rs in splits.items():
        for r in rng.sample(rs, min(args.max_per_split, len(rs))):
            per_image.append({"split": split, "source": r.source, "label": r.label, **image_stats(r.path, size)})

    groups = defaultdict(list)
    for s in per_image:
        groups[(s["source"], s["label"])].append(s)
    report["image_properties"] = {
        f"{src}/label{label}": {
            "n_sampled": len(ss),
            "width": summarize([s["width"] for s in ss]),
            "height": summarize([s["height"] for s in ss]),
            "intensity": summarize([s["intensity"] for s in ss]),
            "contrast": summarize([s["contrast"] for s in ss]),
            "formats": dict(Counter(s["format"] for s in ss)),
        }
        for (src, label), ss in sorted(groups.items())
    }
    # Normalization statistics come from the training split only.
    mean, std = channel_mean_std([s for s in per_image if s["split"] == "train"])
    report["train_normalization"] = {"mean": mean, "std": std}
    report["cross_split_duplicates"] = cross_split_duplicates(splits)

    out = Path(args.out or ("outputs/eda_raw" if args.raw else "outputs/eda"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    with open(out / "per_image.csv", "w", newline="") as f:
        cols = ["split", "source", "label", "width", "height", "format", "mode", "intensity", "contrast"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_image)
    print(json.dumps({k: report[k] for k in ("counts", "train_normalization", "cross_split_duplicates")}, indent=2))
    print(f"full report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
