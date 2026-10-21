"""Exploratory data analysis (manuscript section 3.5): class balance, image properties, leakage."""
import hashlib
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from attribution.data.datasets import Record


def class_balance(records: list[Record]) -> dict:
    """P(y = i) = n_i / N, overall and per source dataset."""
    def dist(rs):
        counts = Counter(r.label for r in rs)
        n = sum(counts.values())
        return {str(k): {"n": v, "p": v / n} for k, v in sorted(counts.items())}

    by_source = defaultdict(list)
    for r in records:
        by_source[r.source].append(r)
    return {"overall": dist(records), "by_source": {s: dist(rs) for s, rs in sorted(by_source.items())}}


def image_stats(path: str, size: int) -> dict:
    """Original size/format plus intensity and contrast measured at the model's input size."""
    with Image.open(path) as img:
        fmt, mode, (w, h) = img.format, img.mode, img.size
        rgb = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float64) / 255.0
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    return {
        "width": w,
        "height": h,
        "format": fmt,
        "mode": mode,
        "intensity": float(luma.mean()),
        "contrast": float(luma.std()),
        "channel_sum": rgb.reshape(-1, 3).sum(0),
        "channel_sq_sum": (rgb.reshape(-1, 3) ** 2).sum(0),
        "pixels": size * size,
    }


def channel_mean_std(stats: list[dict]) -> tuple[list[float], list[float]]:
    """Dataset RGB mean and std for z-score normalization."""
    n = sum(s["pixels"] for s in stats)
    total = sum(s["channel_sum"] for s in stats)
    total_sq = sum(s["channel_sq_sum"] for s in stats)
    mean = total / n
    std = np.sqrt(total_sq / n - mean**2)
    return mean.round(4).tolist(), std.round(4).tolist()


def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def cross_split_duplicates(splits: dict[str, list[Record]]) -> dict[str, int]:
    """Count byte-identical files shared between splits (data leakage check)."""
    seen: dict[str, set[str]] = defaultdict(set)
    for split, rs in splits.items():
        for r in rs:
            seen[file_hash(r.path)].add(split)
    pairs = Counter()
    for split_set in seen.values():
        if len(split_set) > 1:
            pairs["+".join(sorted(split_set))] += 1
    return dict(pairs)
