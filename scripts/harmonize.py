"""Write harmonized copies of the split images (size and JPEG quality from config).

Run after prepare_data.py. Writes data/harmonized/<split>/<n>.jpg and
data/harmonized/<split>.csv pointing at them; training and EDA read these.
"""
import argparse
import csv
from pathlib import Path

from PIL import Image

from attribution.config import load_config
from attribution.data.splits import read_split
from attribution.transforms.harmonize import harmonize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    h = cfg["preprocess"]["harmonize"]
    src, out = Path(cfg["data"]["processed_dir"]), Path(cfg["data"]["harmonized_dir"])

    for split in cfg["data"]["split"]:
        records = read_split(src / f"{split}.csv")
        (out / split).mkdir(parents=True, exist_ok=True)
        rows = []
        for i, r in enumerate(records):
            dst = out / split / f"{i:06d}.jpg"
            with Image.open(r.path) as img:
                harmonize(img, h["size"]).save(dst, "JPEG", quality=h["jpeg_quality"])
            rows.append((str(dst), r.label, r.source, r.path))
        with open(out / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "source", "original_path"])
            w.writerows(rows)
        print(f"{split}: {len(rows)} images -> {out / split}")


if __name__ == "__main__":
    main()
