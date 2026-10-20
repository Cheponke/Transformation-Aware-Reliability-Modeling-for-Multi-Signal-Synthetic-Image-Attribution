"""Index the raw datasets, sample per class, and write train/val/test CSVs."""
import argparse
import csv
from pathlib import Path

from attribution.config import load_config
from attribution.data.datasets import index_dataset
from attribution.data.splits import sample_per_class, stratified_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--datasets", nargs="+", help="dataset names from the config (default: all)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    names = args.datasets or list(d["datasets"])

    records = [r for name in names for r in index_dataset(name, d["datasets"][name]["path"])]
    records = sample_per_class(records, d["samples_per_class"], cfg["seed"])
    splits = stratified_split(records, d["split"], cfg["seed"])

    out = Path(d["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with open(out / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "source"])
            w.writerows((r.path, r.label, r.source) for r in rows)
        print(f"{split}: {len(rows)} images")


if __name__ == "__main__":
    main()
