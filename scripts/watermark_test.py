"""Watermark stress test (manuscript 3.6.1, research question 1): does the mark survive social-media processing?

1. Threshold: score every unmarked validation image under every condition and
   set the detection threshold so at most `target_fpr` of them would be flagged
   under any condition (the detector can't know how an image was processed, and
   falsely flagging human art is the costly error).
2. For each embedding strength alpha, watermark the AI test images
   (I_w = I + alpha * W, as a generator would), leave human images unmarked,
   apply every social-media condition and run the correlation detector.

Reported per alpha and condition: detection rate on marked AI images (how often
the watermark survives), false detection rate on unmarked human images, mean
scores, and the PSNR of the marked images (how visible the mark is).

Writes outputs/watermark/{threshold.json, results.csv, scores.csv}.
Usage: python scripts/watermark_test.py [--max-images 500]
"""
import argparse
import csv
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from attribution.config import load_config, splits_dir
from attribution.data.splits import read_split
from attribution.signals import watermark
from attribution.transforms.social_media import build_conditions

CFG: dict = {}


def _init(cfg: dict) -> None:
    global CFG
    CFG = cfg


def _score_image(job: tuple[str, float | None]) -> list[float] | tuple[list[float], float]:
    """Scores of one image under every condition; alpha=None leaves it unmarked."""
    path, alpha = job
    key = CFG["watermark"]["key"]
    with Image.open(path) as f:
        img = f.convert("RGB")
    marked = watermark.embed(img, alpha, key) if alpha is not None else img
    scores = [watermark.score(t(marked) if t else marked, key) for t in build_conditions(CFG).values()]
    return (scores, watermark.psnr(img, marked)) if alpha is not None else scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-images", type=int, help="per class, for a quick run")
    ap.add_argument("--out", default="outputs/watermark")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    w = cfg["watermark"]
    conditions = list(build_conditions(cfg))
    rng = random.Random(cfg["seed"])

    def subset(records):
        if args.max_images and len(records) > args.max_images:
            return rng.sample(records, args.max_images)
        return records

    val = subset(read_split(splits_dir(cfg) / "val.csv"))
    test = read_split(splits_dir(cfg) / f"{args.split}.csv")
    ai = subset([r for r in test if r.label == 1])
    human = subset([r for r in test if r.label == 0])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        print(f"calibrating threshold on {len(val)} unmarked validation images", flush=True)
        val_scores = np.array(pool.map(_score_image, [(r.path, None) for r in val], chunksize=16))
        per_condition = {name: float(np.quantile(val_scores[:, c], 1 - w["target_fpr"])) for c, name in enumerate(conditions)}
        threshold = max(per_condition.values())
        (out / "threshold.json").write_text(json.dumps(
            {"threshold": threshold, "target_fpr": w["target_fpr"], "key": w["key"], "per_condition": per_condition}, indent=1))
        print(f"threshold = {threshold:.4f} (at most {w['target_fpr']:.0%} of unmarked images above it, under every condition)", flush=True)

        print(f"scoring {len(human)} unmarked human images", flush=True)
        human_scores = np.array(pool.map(_score_image, [(r.path, None) for r in human], chunksize=16))

        rows, per_image = [], []
        for c, name in enumerate(conditions):
            per_image += [(None, name, r.path, 0, round(float(s), 5)) for r, s in zip(human, human_scores[:, c])]
        for alpha in w["alphas"]:
            print(f"alpha={alpha}: scoring {len(ai)} watermarked AI images", flush=True)
            results = pool.map(_score_image, [(r.path, alpha) for r in ai], chunksize=16)
            ai_scores = np.array([s for s, _ in results])
            mean_psnr = float(np.mean([p for _, p in results]))
            for c, name in enumerate(conditions):
                row = {
                    "alpha": alpha,
                    "condition": name,
                    "detection_rate": float((ai_scores[:, c] >= threshold).mean()),
                    "false_detection_rate": float((human_scores[:, c] >= threshold).mean()),
                    "mean_score_marked": float(ai_scores[:, c].mean()),
                    "mean_score_unmarked": float(human_scores[:, c].mean()),
                    "psnr_db": mean_psnr,
                }
                rows.append(row)
                per_image += [(alpha, name, r.path, 1, round(float(s), 5)) for r, s in zip(ai, ai_scores[:, c])]
                print(f"  {name:>14}: detected {row['detection_rate']:.1%}  false alarms {row['false_detection_rate']:.1%}", flush=True)

    with open(out / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(out / "scores.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["alpha", "condition", "path", "label", "score"])  # alpha empty = unmarked
        wr.writerows(per_image)
    print(f"results: {out / 'results.csv'}")


if __name__ == "__main__":
    main()
