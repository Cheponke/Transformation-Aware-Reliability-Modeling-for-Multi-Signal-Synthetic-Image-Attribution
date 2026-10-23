"""Index the raw datasets, sample per class, and write train/val/test CSVs."""
import argparse
import csv
from pathlib import Path

from attribution.config import load_config
from attribution.data.datasets import index_dataset
from attribution.data.eda import file_hash
from attribution.data.splits import deduplicate, sample_per_class, stratified_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--datasets", nargs="+", help="dataset names from the config (default: all)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    names = args.datasets or list(d["datasets"])

    records = []
    for name in names:
        found, unmatched = index_dataset(name, d["datasets"][name])
        print(f"{name}: {len(found)} labeled images, {unmatched} unmatched")
        records.extend(found)
    records = sample_per_class(records, d["samples_per_class"], cfg["seed"], d.get("balance_sources", True))
    for label in (0, 1):
        by_source = {}
        for r in records:
            if r.label == label:
                by_source[r.source] = by_source.get(r.source, 0) + 1
        print(f"sampled label {label}: {by_source}")
    records, removed, conflicts = deduplicate(records, [file_hash(r.path) for r in records])
    print(f"removed {removed} duplicate images ({conflicts} with conflicting labels)")
    splits = stratified_split(records, d["split"], cfg["seed"])
    
###commit
    near_duplicates = cross_split_near_duplicates(splits)
    print(f"Near-duplicate pairs across splits: {len(near_duplicates)}")
    for item in near_duplicates[:20]:
      print(
          f"distance={item['hash_distance']} |"
          f"{item['split_a']} : {item['path_a']}"
          f"{item['split_b']} : {item['path_b']}"
      )

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
