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


# Near-duplicate check (perceptual hash): the same artwork re-saved, resized or
# recompressed has a different file hash but almost the same dHash, so it can
# end up in two splits and leak test images into training.

def perceptual_hash(path: str) -> int:
    """64-bit difference hash (dHash): whether each pixel is brighter than its right neighbour on a 9x8 thumbnail."""
    with Image.open(path) as img:
        gray = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def hash_distance(hash_a: int, hash_b: int) -> int:
    """Hamming distance between two perceptual hashes (number of differing bits)."""
    return (hash_a ^ hash_b).bit_count()


_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between two uint64 hash arrays, shape (len(a), len(b))."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(x).astype(np.uint8)
    return _POPCOUNT8[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1, dtype=np.uint8)


def _hash_one(path: str) -> int | None:
    try:
        return perceptual_hash(path)
    except (OSError, ValueError) as exc:
        print(f"Warning: could not hash {path}: {exc}")
        return None


def hash_records(records: list[Record], workers: int | None = None) -> list[int | None]:
    """Perceptual hash of every record (None where the image can't be read), in parallel when workers > 1."""
    paths = [r.path for r in records]
    if workers is not None and workers <= 1:
        return [_hash_one(p) for p in paths]
    from multiprocessing import Pool

    with Pool(workers) as pool:
        return pool.map(_hash_one, paths, chunksize=64)


def cross_split_near_duplicates(
    splits: dict[str, list[Record]],
    max_distance: int = 5,
    workers: int | None = 1,
    chunk: int = 512,
) -> list[dict]:
    """Pairs of visually similar images (dHash distance <= max_distance) that sit in different splits.

    Distances are computed with numpy, one block of `chunk` images against a
    whole other split at a time, so ~20,000 images take seconds instead of
    hours. Images that can't be read are skipped with a warning.
    """
    if not isinstance(max_distance, int) or not 0 <= max_distance <= 64:
        raise ValueError("max_distance must be an integer between 0 and 64")
    hashed = {}
    for split, records in splits.items():
        hashes = hash_records(records, workers)
        keep = [(r, h) for r, h in zip(records, hashes) if h is not None]
        hashed[split] = ([r for r, _ in keep], np.array([h for _, h in keep], dtype=np.uint64))

    names = list(hashed)
    findings = []
    for i, split_a in enumerate(names):
        rec_a, hash_a = hashed[split_a]
        for split_b in names[i + 1:]:
            rec_b, hash_b = hashed[split_b]
            if len(hash_a) == 0 or len(hash_b) == 0:
                continue
            for start in range(0, len(hash_a), chunk):
                d = _hamming_matrix(hash_a[start:start + chunk], hash_b)
                for ia, ib in zip(*np.nonzero(d <= max_distance)):
                    a, b = rec_a[start + ia], rec_b[ib]
                    findings.append({
                        "split_a": split_a,
                        "path_a": a.path,
                        "label_a": a.label,
                        "source_a": a.source,
                        "split_b": split_b,
                        "path_b": b.path,
                        "label_b": b.label,
                        "source_b": b.source,
                        "hash_distance": int(d[ia, ib]),
                    })
    return findings


def near_duplicate_summary(findings: list[dict], splits: dict[str, list[Record]]) -> dict:
    """Counts for the report: pairs per split pair, label agreement, and images affected per split."""
    by_pair = Counter(f"{f['split_a']}+{f['split_b']}" for f in findings)
    affected = {s: set() for s in splits}
    for f in findings:
        affected[f["split_a"]].add(f["path_a"])
        affected[f["split_b"]].add(f["path_b"])
    return {
        "n_pairs": len(findings),
        "pairs_by_split": dict(sorted(by_pair.items())),
        "pairs_same_label": sum(f["label_a"] == f["label_b"] for f in findings),
        "pairs_conflicting_label": sum(f["label_a"] != f["label_b"] for f in findings),
        "pairs_by_source": dict(sorted(Counter("+".join(sorted((f["source_a"], f["source_b"]))) for f in findings).items())),
        "pairs_by_distance": {str(k): v for k, v in sorted(Counter(f["hash_distance"] for f in findings).items())},
        "images_affected": {s: len(p) for s, p in affected.items()},
        "split_sizes": {s: len(r) for s, r in splits.items()},
    }


NEAR_DUPLICATE_COLUMNS = ["split_a", "path_a", "label_a", "source_a", "split_b", "path_b", "label_b", "source_b", "hash_distance"]


def write_near_duplicate_report(findings: list[dict], summary: dict, out_dir) -> None:
    """Write near_duplicates.csv (every pair) and near_duplicates_summary.json, and print the headline counts."""
    import csv
    import json
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "near_duplicates.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NEAR_DUPLICATE_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(findings, key=lambda x: (x["hash_distance"], x["split_a"], x["path_a"])))
    (out_dir / "near_duplicates_summary.json").write_text(json.dumps(summary, indent=1))

    print(f"Near-duplicate pairs across splits: {summary['n_pairs']} "
          f"({summary['pairs_same_label']} same label, {summary['pairs_conflicting_label']} conflicting labels)")
    print(f"  by split pair: {summary['pairs_by_split']}")
    print(f"  images involved per split: {summary['images_affected']} of {summary['split_sizes']}")
    print(f"  by source dataset: {summary['pairs_by_source']}")
    print(f"  by hash distance: {summary['pairs_by_distance']}")
    for item in sorted(findings, key=lambda x: x["hash_distance"])[:20]:
        print(f"  distance={item['hash_distance']} | {item['split_a']}: {item['path_a']} | {item['split_b']}: {item['path_b']}")
    print(f"Report: {out_dir / 'near_duplicates.csv'}, {out_dir / 'near_duplicates_summary.json'}")
