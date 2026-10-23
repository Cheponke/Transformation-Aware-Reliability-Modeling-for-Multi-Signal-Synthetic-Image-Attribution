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
    
### commit
def perceptual_hash(path: str) -> int:
    """Compute a 64-bit difference hash (dHash) for an image."""
    with Image.open(path) as img:
        gray = img.convert("L").resize(
            (9, 8),
            Image.Resampling.LANCZOS,
        )
        pixels = np.asarray(gray, dtype=np.int16)
    differences = pixels[:, 1:] > pixels[:, :-1] 
    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)
    return value

def hash_distance(hash_a: int, hash_b: int) -> int:
    """Calculate the Hamming distance between two perceptual hashes."""
    return (hash_a ^ hash_b).bit_count()

def cross_split_near_duplicates(
    splits: dict[str, list[Record]],
    max_distance: int = 5,
) -> list[dict]:
    """Find visually similar images appearing in different data splits."""
    if not isinstance(max_distance, int) or not 0 <= max_distance <= 64:
        raise ValueError("max_distance must be an integer between 0 and 64")
    entries = []
    for split, records in splits.items():
        for record in records:
            try:
                image_hash = perceptual_hash(record.path)
                entries.append({
                    "split": split,
                    "path": record.path,
                    "label": record.label,
                    "hash": image_hash,
                })
            except (OSError, ValueError) as exc:
                print(
                    f"Warning: Could not hash {record.path}: {exc}"
                )
    findings = []
    for i, first in enumerate(entries):
        for second in entries[i + 1:]:
            if first["split"] == second["split"]:
                continue
            distance = hash_distance(
                first["hash"],
                second["hash"],
            )
            if distance <= max_distance:
                findings.append({
                    "split_a": first["split"],
                    "path_a": first["path"],
                    "label_a": first["label"],
                    "split_b": second["split"],
                    "path_b": second["path"],
                    "label_b": second["label"],
                    "hash_distance": distance,
                })
    return findings
