"""Remove the leaking copy of each cross-split near-duplicate pair from the split CSVs.

Reads the pair report from scripts/check_near_duplicates.py and, for every pair,
drops the copy in the lower-priority split: a train copy always, a val copy
only when its partner is in test. The test set is left unchanged, so results
before and after retraining are measured on exactly the same test images.

Rewrites <harmonized_dir>/<split>.csv and <processed_dir>/<split>.csv in place
(the image files are left alone; they are simply no longer listed) and writes
<report-dir>/removed_near_duplicates.csv. Running it again is safe: images
that are already gone are skipped.

Usage: python scripts/remove_near_duplicates.py --pairs outputs/data_quality/near_duplicates.csv
"""
import argparse
import csv
import json
import os
from pathlib import Path

from attribution.config import load_config
from attribution.data.splits import leaking_partners


def _read(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pairs", default="outputs/data_quality/near_duplicates.csv")
    ap.add_argument("--report-dir", default="outputs/data_quality")
    args = ap.parse_args()
    cfg = load_config(args.config)
    harmonized, processed = Path(cfg["data"]["harmonized_dir"]), Path(cfg["data"]["processed_dir"])

    with open(args.pairs, newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    drop = leaking_partners(pairs)
    removed, summary = [], {}
    for split in cfg["data"]["split"]:
        fields, rows = _read(harmonized / f"{split}.csv")
        to_drop = drop.get(split, set())
        kept = [r for r in rows if os.path.normpath(r["path"]) not in to_drop]
        gone = [r for r in rows if os.path.normpath(r["path"]) in to_drop]
        _write(harmonized / f"{split}.csv", fields, kept)
        # the pre-harmonization CSV lists the same images by their original path
        processed_csv = processed / f"{split}.csv"
        if processed_csv.is_file() and gone and "original_path" in fields:
            originals = {os.path.normpath(r["original_path"]) for r in gone}
            p_fields, p_rows = _read(processed_csv)
            _write(processed_csv, p_fields, [r for r in p_rows if os.path.normpath(r["path"]) not in originals])
        removed += [{"split": split, **r} for r in gone]
        labels = [int(r["label"]) for r in kept]
        summary[split] = {"before": len(rows), "removed": len(gone), "after": len(kept),
                          "human_after": labels.count(0), "ai_after": labels.count(1)}
        print(f"{split:>5}: {len(rows)} -> {len(kept)} images ({len(gone)} removed; "
              f"now {labels.count(0)} human / {labels.count(1)} AI)")

    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = out / "removed_near_duplicates.csv"
    if removed:
        _write(report, list(removed[0]), removed)
        (out / "removed_near_duplicates_summary.json").write_text(json.dumps(
            {"pairs_file": args.pairs, "n_pairs": len(pairs), "rule": "keep test > val > train", "splits": summary}, indent=1))
        print(f"Removed {len(removed)} images; list written to {report}")
    else:
        print("Nothing to remove: every flagged leaking copy is already gone from the split CSVs.")


if __name__ == "__main__":
    main()
