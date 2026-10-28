"""Near-duplicate check on the existing splits (data leakage, manuscript section 3.5).

Reads the split CSVs the experiments already use (data/harmonized/ when
harmonization is on) and reports pairs of visually similar images that sit in
different splits: the same artwork re-saved, resized or recompressed, which a
file hash misses. Nothing is changed or removed; the report is for deciding
whether the splits need rebuilding.

Writes <out>/near_duplicates.csv (every pair, closest first) and
<out>/near_duplicates_summary.json (counts by split pair, label agreement,
distance, and images affected per split).
Usage: python scripts/check_near_duplicates.py [--max-distance 5]
"""
import argparse

from attribution.config import load_config, splits_dir
from attribution.data.eda import cross_split_near_duplicates, near_duplicate_summary, write_near_duplicate_report
from attribution.data.splits import read_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-distance", type=int, default=5, help="dHash bits that may differ for a near-duplicate")
    ap.add_argument("--out", default="outputs/data_quality")
    ap.add_argument("--workers", type=int, default=None, help="processes for hashing (default: all cores)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    folder = splits_dir(cfg)
    splits = {name: read_split(folder / f"{name}.csv") for name in cfg["data"]["split"]}
    print(f"Checking {sum(map(len, splits.values()))} images in {folder} ({', '.join(f'{k}: {len(v)}' for k, v in splits.items())})")
    findings = cross_split_near_duplicates(splits, args.max_distance, workers=args.workers)
    summary = near_duplicate_summary(findings, splits)
    summary["max_distance"] = args.max_distance
    summary["splits_dir"] = str(folder)
    write_near_duplicate_report(findings, summary, args.out)


if __name__ == "__main__":
    main()
