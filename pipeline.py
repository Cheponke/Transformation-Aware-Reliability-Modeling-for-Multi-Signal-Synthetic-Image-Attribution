"""ResNet-18 compression baseline, start to finish, in one file.

    python pipeline.py                    # everything: download -> splits -> EDA -> train -> stress test
    python pipeline.py --arch resnet50    # another backbone on the same prepared data
    python pipeline.py --from train       # re-run training and everything after it

Needs: pip install torch torchvision pillow numpy kaggle
(RTX 50-series: install torch from https://download.pytorch.org/whl/cu128)
and a Kaggle token at ~/.kaggle/kaggle.json.

Steps whose output already exists are skipped, so after a crash or disconnect
just run the same command again. Everything is written under data/ and outputs/.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------------- settings
# Taken from the accepted manuscript (Chapter 3); change here, nowhere else.
SEED = 42
RAW, SPLITS, OUT = Path("data/raw"), Path("data/harmonized"), Path("outputs")
HUMAN, AI = 0, 1

# Each dataset's Kaggle slug and how its folder names map to labels (first match wins).
DATASETS = {
    "kannan_ai_human_art": {  # labeled train split only; its test/ folder is unlabeled
        "kaggle": "kausthubkannan/ai-and-human-art-classification",
        "classes": [("*NON_AI*", HUMAN), ("*AI_GENERATED*", AI)],
    },
    "superpotato9_ai_recognition": {
        "kaggle": "superpotato9/dalle-recognition-dataset",
        "classes": [("*real*", HUMAN), ("*fake*", AI)],
    },
    "ai_artbench": {  # style folders like AI_SD_baroque are AI, the rest human
        "kaggle": "ravidussilva/real-ai-art",
        "classes": [("*/AI_*", AI), ("*", HUMAN)],
    },
}
FOLDER = {"kannan_ai_human_art": "kannan", "superpotato9_ai_recognition": "superpotato9", "ai_artbench": "ai_artbench"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SAMPLES_PER_CLASS = 10_000          # split evenly across the three datasets
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
HARMONIZE_SIZE, HARMONIZE_QUALITY = 256, 95   # every image -> 256 px JPEG q95, removes format/size bias

IMAGE_SIZE = 224
MEAN, STD = [0.4827, 0.4395, 0.3883], [0.2823, 0.2705, 0.2714]   # from the EDA on the training split
AUG = {"rotation": 15, "brightness": 0.2, "contrast": 0.2, "jitter": 0.1}

LR, WEIGHT_DECAY, BATCH_SIZE, EPOCHS, PATIENCE, DROPOUT = 1e-4, 1e-4, 32, 50, 5, 0.5
NUM_WORKERS = min(4, os.cpu_count() or 1)

# Simulated social-media processing for the stress test.
CONDITIONS = [("clean", None, None), ("jpeg_q90", "jpeg", 90), ("jpeg_q70", "jpeg", 70), ("jpeg_q50", "jpeg", 50),
              ("jpeg_q30", "jpeg", 30), ("webp_q80", "webp", 80), ("resize_0.5", "resize", 0.5),
              ("resize_0.25", "resize", 0.25), ("recompress_x3", "recompress", 3)]
RECOMPRESS_QUALITY, ECE_BINS = 75, 15


# ---------------------------------------------------------------- 1. download
def download() -> None:
    for name, spec in DATASETS.items():
        out = RAW / FOLDER[name]
        if (out / ".download_complete").exists():
            print(f"{name}: already downloaded")
            continue
        shutil.rmtree(out, ignore_errors=True)  # a half-finished download starts over
        out.mkdir(parents=True)
        print(f"{name}: downloading {spec['kaggle']} (this can take a while)", flush=True)
        subprocess.run(["kaggle", "datasets", "download", "-d", spec["kaggle"], "-p", str(out), "--unzip"], check=True)
        (out / ".download_complete").touch()


# ---------------------------------------------------------------- 2. label, sample, split, harmonize
def label_for(rel_dir: str, rules) -> int | None:
    return next((label for pattern, label in rules if fnmatch(rel_dir, pattern)), None)


def index_all() -> list[tuple[str, int, str]]:
    """(path, label, dataset) for every labeled image; prints how each folder was labeled."""
    records = []
    for name, spec in DATASETS.items():
        root = RAW / FOLDER[name]
        folders, unmatched = Counter(), 0
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = p.parent.relative_to(root).as_posix()
            label = label_for(rel, spec["classes"])
            folders[(rel, label)] += 1
            if label is None:
                unmatched += 1
            else:
                records.append((str(p), label, name))
        print(f"{name}: {sum(folders.values()) - unmatched} labeled images, {unmatched} without a label")
        for (rel, label), n in sorted(folders.items()):
            print(f"   {n:>7}  {({HUMAN: 'human', AI: 'ai'}).get(label, 'none'):<6} {rel}")
    return records


def sample_per_class(records, n: int, rng: random.Random):
    """n images per class, split evenly across datasets; a small dataset gives all it has."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups[r[1]][r[2]].append(r)
    out = []
    for label in sorted(groups):
        by_size = sorted(groups[label].items(), key=lambda kv: (len(kv[1]), kv[0]))
        remaining = n
        for i, (_, items) in enumerate(by_size):
            take = min(remaining // (len(by_size) - i), len(items))
            out.extend(rng.sample(items, take))
            remaining -= take
    return out


def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def deduplicate(records):
    """One copy per byte-identical image, so nothing can land in two splits; label conflicts are dropped."""
    by_hash = defaultdict(list)
    for r in records:
        by_hash[file_hash(r[0])].append(r)
    kept = [copies[0] for copies in by_hash.values() if len({c[1] for c in copies}) == 1]
    print(f"removed {len(records) - len(kept)} duplicate images")
    return kept


def stratified_split(records, rng: random.Random) -> dict[str, list]:
    splits = {name: [] for name in SPLIT_RATIOS}
    for label in (HUMAN, AI):
        items = [r for r in records if r[1] == label]
        rng.shuffle(items)
        start = 0
        for i, (name, ratio) in enumerate(SPLIT_RATIOS.items()):
            end = len(items) if i == len(SPLIT_RATIOS) - 1 else start + round(ratio * len(items))
            splits[name].extend(items[start:end])
            start = end
    return splits


def harmonize(img: Image.Image) -> Image.Image:
    """Shorter side to 256 px, center-crop square, RGB (transparency flattened onto white)."""
    if img.mode in ("P", "LA", "RGBA"):
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    return ImageOps.fit(img.convert("RGB"), (HARMONIZE_SIZE, HARMONIZE_SIZE), Image.BICUBIC)


def prepare() -> None:
    rng = random.Random(SEED)
    records = deduplicate(sample_per_class(index_all(), SAMPLES_PER_CLASS, rng))
    for split, rows in stratified_split(records, rng).items():
        (SPLITS / split).mkdir(parents=True, exist_ok=True)
        out_rows = []
        for i, (path, label, source) in enumerate(rows):
            dst = SPLITS / split / f"{i:06d}.jpg"
            with Image.open(path) as img:
                harmonize(img).save(dst, "JPEG", quality=HARMONIZE_QUALITY)
            out_rows.append((dst.as_posix(), label, source, path))
        with open(SPLITS / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "source", "original_path"])
            w.writerows(out_rows)
        print(f"{split}: {len(out_rows)} images -> {SPLITS / split}", flush=True)


def read_split(split: str) -> list[tuple[str, int, str]]:
    with open(SPLITS / f"{split}.csv") as f:
        return [(r["path"], int(r["label"]), r["source"]) for r in csv.DictReader(f)]


# ---------------------------------------------------------------- 3. EDA
def eda(max_per_split: int = 2000) -> None:
    rng = random.Random(SEED)
    splits = {s: read_split(s) for s in SPLIT_RATIOS}
    report = {"counts": {s: len(rs) for s, rs in splits.items()}, "class_balance": {}}
    for s, rs in splits.items():
        report["class_balance"][s] = {f"{src}/{'ai' if lab else 'human'}": n
                                      for (src, lab), n in sorted(Counter((r[2], r[1]) for r in rs).items())}
    rows, total, total_sq, pixels = [], np.zeros(3), np.zeros(3), 0
    for s, rs in splits.items():
        for path, label, source in rng.sample(rs, min(max_per_split, len(rs))):
            with Image.open(path) as img:
                rgb = np.asarray(img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR), float) / 255
            luma = rgb @ np.array([0.299, 0.587, 0.114])
            rows.append({"split": s, "source": source, "label": label,
                         "intensity": float(luma.mean()), "contrast": float(luma.std())})
            if s == "train":  # normalization comes from the training split only
                total += rgb.reshape(-1, 3).sum(0)
                total_sq += (rgb.reshape(-1, 3) ** 2).sum(0)
                pixels += IMAGE_SIZE * IMAGE_SIZE
    mean = total / pixels
    report["train_normalization"] = {"mean": mean.round(4).tolist(),
                                     "std": np.sqrt(total_sq / pixels - mean**2).round(4).tolist()}
    seen = defaultdict(set)
    for s, rs in splits.items():
        for r in rs:
            seen[file_hash(r[0])].add(s)
    report["cross_split_duplicates"] = dict(Counter("+".join(sorted(v)) for v in seen.values() if len(v) > 1))
    (OUT / "eda").mkdir(parents=True, exist_ok=True)
    (OUT / "eda" / "report.json").write_text(json.dumps(report, indent=2))
    with open(OUT / "eda" / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: report[k] for k in ("counts", "train_normalization", "cross_split_duplicates")}, indent=2))
    print(f"if train_normalization differs from MEAN/STD at the top of this file, copy it there")


# ---------------------------------------------------------------- 4. train
def build_model(arch: str, pretrained: bool):
    import torch.nn as nn
    from torchvision import models
    builders = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    }
    builder, weights = builders[arch]
    model = builder(weights=weights if pretrained else None)
    head = lambda n: nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(n, 1))  # single logit: P(AI)
    if arch.startswith("resnet"):
        model.fc = head(model.fc.in_features)
    else:
        model.classifier = head(model.classifier[-1].in_features)
    return model


class Degrade:
    """One social-media condition; a class (not a lambda) so DataLoader workers can use it on Windows."""

    def __init__(self, kind, value):
        self.kind, self.value = kind, value

    @staticmethod
    def encode(img, fmt, quality):
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def __call__(self, img):
        if self.kind == "jpeg":
            return self.encode(img, "JPEG", self.value)
        if self.kind == "webp":
            return self.encode(img, "WEBP", self.value)
        if self.kind == "resize":
            w, h = img.size
            return img.resize((max(1, round(w * self.value)), max(1, round(h * self.value))), Image.BICUBIC)
        for _ in range(self.value):  # recompress: repeated download/repost cycles
            img = self.encode(img, "JPEG", RECOMPRESS_QUALITY)
        return img


class ImageSet:
    """(image tensor, label) pairs; top-level so DataLoader workers can load it on Windows too."""

    def __init__(self, records, tf, degrade=None):
        self.records, self.tf, self.degrade = records, tf, degrade

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        import torch
        path, label, _ = self.records[i]
        img = Image.open(path).convert("RGB")
        if self.degrade is not None:
            img = self.degrade(img)
        return self.tf(img), torch.tensor(label, dtype=torch.float32)


def loader(records, train: bool, degrade=None, shuffle=False):
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    ops = [T.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        ops += [T.RandomRotation(AUG["rotation"]), T.RandomHorizontalFlip(),
                T.ColorJitter(AUG["brightness"], AUG["contrast"], AUG["jitter"], AUG["jitter"] / 2)]
    tf = T.Compose(ops + [T.ToTensor(), T.Normalize(MEAN, STD)])
    return DataLoader(ImageSet(records, tf, degrade), batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)


def predict(model, dl, device):
    import torch
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, y in dl:
            logits.extend(model(x.to(device)).squeeze(1).float().cpu().tolist())
            labels.extend(y.int().tolist())
    return np.array(logits), np.array(labels)


def bce(logits, labels, temperature=1.0) -> float:
    p = np.clip(1 / (1 + np.exp(-logits / temperature)), 1e-12, 1 - 1e-12)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def fit_temperature(logits, labels) -> float:
    """Temperature scaling: golden-section search for T minimizing validation NLL."""
    a, b, g = np.log(0.05), np.log(20.0), (np.sqrt(5) - 1) / 2
    for _ in range(100):
        c, d = b - g * (b - a), a + g * (b - a)
        if bce(logits, labels, np.exp(c)) < bce(logits, labels, np.exp(d)):
            b = d
        else:
            a = c
    return float(np.exp((a + b) / 2))


def train(arch: str) -> None:
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = OUT / "models" / arch
    out.mkdir(parents=True, exist_ok=True)
    train_dl = loader(read_split("train"), train=True, shuffle=True)
    val_dl = loader(read_split("val"), train=False)
    model = build_model(arch, pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    amp = device == "cuda"  # mixed precision roughly doubles GPU throughput
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    print(f"training {arch} on {device}: {len(train_dl.dataset)} train / {len(val_dl.dataset)} val images", flush=True)

    best, stale, history = float("inf"), 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=amp):
                loss = loss_fn(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
            n += len(y)
        logits, labels = predict(model, val_dl, device)
        val_loss, val_acc = bce(logits, labels), float(((logits > 0) == labels).mean())
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch}: train_loss={total / n:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=1))
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"early stop: no validation improvement for {PATIENCE} epochs")
                break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    temperature = fit_temperature(*predict(model, val_dl, device))
    (out / "calibration.json").write_text(json.dumps({"temperature": temperature}))
    print(f"saved {out / 'best.pt'}, temperature={temperature:.3f}")


# ---------------------------------------------------------------- 5. stress test
def metrics(labels, probs) -> dict:
    pred = (probs >= 0.5).astype(int)
    tp, tn = int(((labels == 1) & (pred == 1)).sum()), int(((labels == 0) & (pred == 0)).sum())
    fp, fn = int(((labels == 0) & (pred == 1)).sum()), int(((labels == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    conf, correct = np.maximum(probs, 1 - probs), (pred == labels).astype(float)
    ece, edges = 0.0, np.linspace(0.5, 1.0, ECE_BINS + 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return {"accuracy": (tp + tn) / len(labels), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,  # human art wrongly flagged as AI
            "ece": float(ece)}


def stress_test(arch: str) -> None:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = OUT / "models" / arch
    model = build_model(arch, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device))
    temperature = json.loads((model_dir / "calibration.json").read_text())["temperature"]
    records = read_split("test")
    rows = []
    for name, kind, value in CONDITIONS:
        logits, labels = predict(model, loader(records, train=False, degrade=Degrade(kind, value) if kind else None), device)
        row = {"condition": name, **metrics(labels, 1 / (1 + np.exp(-logits / temperature)))}
        rows.append(row)
        print(f"{name:>14}: acc={row['accuracy']:.4f} f1={row['f1']:.4f} fpr={row['fpr']:.4f} ece={row['ece']:.4f}", flush=True)
    for row in rows:
        row["delta_a"] = rows[0]["accuracy"] - row["accuracy"]  # delta-A = A_clean - A_condition
    out = OUT / "stress_test" / arch
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"results: {out / 'results.csv'}")


# ---------------------------------------------------------------- run
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50", "efficientnet_b0"])
    ap.add_argument("--from", dest="start", choices=["download", "prepare", "eda", "train", "stress_test"],
                    help="re-run this step and every step after it")
    args = ap.parse_args()

    # (name, function, file that exists once the step is done)
    steps = [
        ("download", download, None),  # skips finished datasets by itself
        ("prepare", prepare, SPLITS / "test.csv"),
        ("eda", eda, OUT / "eda" / "report.json"),
        ("train", lambda: train(args.arch), OUT / "models" / args.arch / "calibration.json"),
        ("stress_test", lambda: stress_test(args.arch), OUT / "stress_test" / args.arch / "results.csv"),
    ]
    rerun = [s[0] for s in steps].index(args.start) if args.start else len(steps)
    for i, (name, fn, done) in enumerate(steps):
        if i < rerun and done is not None and done.exists():
            print(f"== {name}: already done, skipping", flush=True)
            continue
        print(f"\n== {name}", flush=True)
        fn()
        if done is not None:  # later steps depend on this one, so they re-run too
            rerun = min(rerun, i)
    print(f"\nall done: {OUT / 'stress_test' / args.arch / 'results.csv'}")


if __name__ == "__main__":
    main()
