"""5-fold cross-validation with paired t-tests (manuscript section 3.8.1).

The pool is every image in the current train/val/test splits. Near-duplicates
are grouped across the whole pool (any chain of dHash distance <= 5) and each
group stays inside one fold, so no copy of a test image is ever trained on.
Folds are balanced by class.

For each fold k (resumable: finished steps are skipped):
  1. test = fold k; the other folds are split into train and validation
     (about 15% of each class held out for validation, groups kept whole);
  2. each backbone is trained on train and temperature-calibrated on val;
  3. the watermark threshold (at most target_fpr of unmarked val images flagged
     under any condition) and its confidence P(AI | detected) are set on val;
  4. every test image is evaluated under every condition by all four systems
     (provenance only, watermark only, CNN only, multi-signal fusion), with
     both backbones reading the same delivered image.

Then, across folds: mean and standard deviation of every metric, and for each
backbone and condition a paired t-test of multi-signal F1 against each
single-signal F1 (5 pairs, one per fold), with Holm-adjusted p-values for the
whole family of tests alongside the unadjusted ones the manuscript specifies.

Writes outputs/cv/: folds.csv, fold_<k>/..., summary_by_fold.csv,
summary_mean_sd.csv, significance.csv, cv_report.json.
Usage: python scripts/cross_validate.py [--folds 0 1] [--archs resnet50 efficientnet_b0] [--aggregate-only]
"""
import argparse
import csv
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attribution.config import load_config, splits_dir  # noqa: E402
from attribution.data.datasets import Record  # noqa: E402
from attribution.data.eda import near_duplicate_groups  # noqa: E402
from attribution.data.splits import grouped_holdout, grouped_k_folds, read_split  # noqa: E402
from attribution.evaluation.statistics import holm_adjust, paired_f1_test  # noqa: E402
from attribution.signals import watermark  # noqa: E402
from attribution.transforms.social_media import build_conditions  # noqa: E402

SINGLE_SYSTEMS = ["provenance_only", "watermark_only", "cnn_only"]
REPORT_METRICS = ["coverage", "accuracy", "precision", "recall", "fpr", "f1", "ece"]


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _records(rows: list[dict]) -> list[Record]:
    return [Record(r["path"], int(r["label"]), r["source"]) for r in rows]


# ---------------------------------------------------------------- folds

def make_folds(cfg: dict, out: Path, k: int, max_distance: int, workers: int | None) -> list[dict]:
    """Assign every image in the current splits to a fold, keeping near-duplicate groups together."""
    path = out / "folds.csv"
    if path.is_file():
        rows = _read_csv(path)
        print(f"folds: reusing {path} ({len(rows)} images)")
        return rows
    folder = splits_dir(cfg)
    pool = [r for name in cfg["data"]["split"] for r in read_split(folder / f"{name}.csv")]
    print(f"folds: grouping near-duplicates among {len(pool)} images (dHash distance <= {max_distance})", flush=True)
    groups = near_duplicate_groups(pool, max_distance, workers=workers)
    fold_of = grouped_k_folds(pool, groups, k, int(cfg["seed"]))
    rows = [{"path": r.path, "label": r.label, "source": r.source, "fold": f, "group": g}
            for r, f, g in zip(pool, fold_of, groups)]
    _write_csv(path, rows)
    sizes = np.bincount(groups)
    print(f"folds: {len(sizes)} groups, {int((sizes > 1).sum())} with more than one image "
          f"({int(sizes[sizes > 1].sum())} images), largest {int(sizes.max())}")
    for f in range(k):
        labels = [r.label for r, x in zip(pool, fold_of) if x == f]
        print(f"  fold {f}: {len(labels)} images ({labels.count(0)} human / {labels.count(1)} AI)")
    return rows


def fold_splits(fold_rows: list[dict], k_fold: int, val_fraction: float, seed: int, fold_dir: Path) -> dict[str, list[Record]]:
    """train/val/test records for one fold (written to fold_dir/{train,val,test}.csv the first time)."""
    if all((fold_dir / f"{s}.csv").is_file() for s in ("train", "val", "test")):
        return {s: read_split(fold_dir / f"{s}.csv") for s in ("train", "val", "test")}
    test = [r for r in fold_rows if int(r["fold"]) == k_fold]
    rest = [r for r in fold_rows if int(r["fold"]) != k_fold]
    train, val = grouped_holdout(_records(rest), [int(r["group"]) for r in rest], val_fraction, seed + k_fold)
    splits = {"train": train, "val": val, "test": _records(test)}
    for name, recs in splits.items():
        _write_csv(fold_dir / f"{name}.csv", [{"path": r.path, "label": r.label, "source": r.source} for r in recs])
    return splits


# ---------------------------------------------------------------- training

def train_backbone(cfg: dict, arch: str, splits: dict, out_dir: Path, seed: int, device: str, epochs: int | None) -> None:
    import torch
    from torch.utils.data import DataLoader

    from attribution.data.torch_dataset import ImageRecords, eval_transform, train_transform
    from attribution.models.calibration import fit_temperature
    from attribution.models.cnn import build_model
    from attribution.models.train import train
    from attribution.signals.forensic import predict_logits

    if (out_dir / "calibration.json").is_file():
        print(f"  {arch}: already trained", flush=True)
        return
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    bs, workers = cfg["train"]["batch_size"], cfg["train"]["num_workers"]
    train_loader = DataLoader(ImageRecords(splits["train"], train_transform(cfg)), batch_size=bs, shuffle=True,
                              num_workers=workers, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(ImageRecords(splits["val"], eval_transform(cfg)), batch_size=bs, num_workers=workers)
    model = build_model(arch, cfg["model"]["pretrained"], cfg["model"]["dropout"])
    print(f"  {arch}: training on {device}: {len(splits['train'])} train / {len(splits['val'])} val images", flush=True)
    best = train(model, train_loader, val_loader, cfg, out_dir, device, epochs)
    model.load_state_dict(torch.load(best, map_location=device))
    logits, labels = predict_logits(model, val_loader, device)
    temperature = fit_temperature(logits, labels)
    (out_dir / "calibration.json").write_text(json.dumps({"temperature": temperature, "seed": seed}))
    print(f"  {arch}: temperature={temperature:.3f}", flush=True)


# ---------------------------------------------------------------- watermark calibration

_WM: dict = {}


def _wm_init(cfg: dict) -> None:
    _WM["cfg"] = cfg


def _wm_scores(job: tuple[str, float | None]) -> list[float]:
    from PIL import Image

    path, alpha = job
    cfg = _WM["cfg"]
    key = int(cfg["watermark"]["key"])
    with Image.open(path) as f:
        img = f.convert("RGB")
    if alpha is not None:
        img = watermark.embed(img, alpha, key)
    return [watermark.score(t(img) if t else img, key) for t in build_conditions(cfg).values()]


def calibrate_watermark(cfg: dict, val: list[Record], out_path: Path, workers: int | None) -> dict:
    """Threshold on unmarked val images (as notebook 03) and P(AI | detected) (as evaluate.py), for one fold."""
    if out_path.is_file():
        return json.loads(out_path.read_text())
    w = cfg["watermark"]
    alpha = float(w["alpha"])
    with Pool(workers, initializer=_wm_init, initargs=(cfg,)) as pool:
        unmarked = np.array(pool.map(_wm_scores, [(r.path, None) for r in val], chunksize=16))
        ai = [r for r in val if r.label == 1]
        marked = np.array(pool.map(_wm_scores, [(r.path, alpha) for r in ai], chunksize=16))
    threshold = float(max(np.quantile(unmarked[:, c], 1 - float(w["target_fpr"])) for c in range(unmarked.shape[1])))
    human = unmarked[[i for i, r in enumerate(val) if r.label == 0]]
    tp, fp = int((marked >= threshold).sum()), int((human >= threshold).sum())
    tpr = (tp + 0.5) / (marked.size + 1)
    fpr = (fp + 0.5) / (human.size + 1)
    result = {"threshold": threshold, "confidence": tpr / (tpr + fpr), "tpr": tp / marked.size, "fpr": fp / human.size,
              "alpha": alpha, "n_val": len(val), "target_fpr": float(w["target_fpr"])}
    out_path.write_text(json.dumps(result, indent=1))
    return result


# ---------------------------------------------------------------- evaluation

def evaluate_fold(cfg: dict, archs: list[str], splits: dict, fold_dir: Path, wm: dict, device: str,
                  batch_size: int, use_provenance: bool) -> None:
    import torch

    import evaluate as ev
    from attribution.models.cnn import build_model
    from attribution.signals import provenance as prov

    summary_path = fold_dir / "summary.csv"
    if summary_path.is_file():
        print("  evaluation: already done", flush=True)
        return
    models = {}
    for arch in archs:
        model = build_model(arch, pretrained=False, dropout=cfg["model"]["dropout"])
        model.load_state_dict(torch.load(fold_dir / arch / "best.pt", map_location=device))
        model.to(device).eval()
        temperature = json.loads((fold_dir / arch / "calibration.json").read_text())["temperature"]
        models[arch] = (model, float(temperature))
    signer = ctx = None
    if use_provenance:
        creds = prov.make_credentials()
        signer, ctx = prov.signer(creds), prov.context([creds.root_pem])

    rows = []
    for condition in build_conditions(cfg):
        print(f"  evaluation: {condition}", flush=True)
        rows += ev.prediction_rows_multi(splits["test"], condition, cfg, models, device, wm["threshold"],
                                         wm["confidence"], signer, ctx, use_provenance, batch_size)
    _write_csv(fold_dir / "per_image.csv", rows)
    ece_bins = int(cfg["evaluation"]["ece_bins"])
    summary = []
    for arch in archs:
        for item in ev.final_summary([r for r in rows if r["arch"] == arch], ece_bins):
            summary.append({"arch": arch, **item})
    _write_csv(summary_path, summary)


# ---------------------------------------------------------------- aggregation

def _num(x):
    if x in (None, "", "None"):
        return None
    return float(x)


def _f1_for_test(item: dict) -> float:
    """F1 for the t-test: a system that flagged no AI image has F1 = 0 (precision undefined, recall 0)."""
    f1 = _num(item.get("f1"))
    return 0.0 if f1 is None else f1


def aggregate(out: Path, k: int, alpha: float) -> dict:
    by_fold = []
    done = []
    for f in range(k):
        path = out / f"fold_{f}" / "summary.csv"
        if path.is_file():
            done.append(f)
            by_fold += [{"fold": f, **row} for row in _read_csv(path)]
    if not by_fold:
        raise RuntimeError(f"no finished folds in {out}")
    _write_csv(out / "summary_by_fold.csv", by_fold)

    keys = list(dict.fromkeys((r["arch"], r["condition"], r["system"]) for r in by_fold))
    mean_sd = []
    for arch, condition, system in keys:
        rows = [r for r in by_fold if (r["arch"], r["condition"], r["system"]) == (arch, condition, system)]
        item = {"arch": arch, "condition": condition, "system": system, "n_folds": len(rows)}
        for m in REPORT_METRICS:
            values = [v for v in (_num(r.get(m)) for r in rows) if v is not None]
            item[f"{m}_mean"] = float(np.mean(values)) if values else None
            item[f"{m}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        mean_sd.append(item)
    _write_csv(out / "summary_mean_sd.csv", mean_sd)

    tests = []
    if len(done) >= 2:
        for arch, condition in dict.fromkeys((a, c) for a, c, _ in keys):
            fusion = {int(r["fold"]): _f1_for_test(r) for r in by_fold
                      if (r["arch"], r["condition"], r["system"]) == (arch, condition, "multi_signal")}
            for single in SINGLE_SYSTEMS:
                other = {int(r["fold"]): _f1_for_test(r) for r in by_fold
                         if (r["arch"], r["condition"], r["system"]) == (arch, condition, single)}
                folds = sorted(set(fusion) & set(other))
                a, b = [fusion[f] for f in folds], [other[f] for f in folds]
                diffs = np.subtract(a, b)
                row = {"arch": arch, "condition": condition, "comparison": f"multi_signal vs {single}",
                       "n_folds": len(folds), "mean_f1_multi_signal": float(np.mean(a)),
                       f"mean_f1_single": float(np.mean(b)), "mean_difference": float(np.mean(diffs)),
                       "sd_difference": float(np.std(diffs, ddof=1)), "t_statistic": None, "p_value": None, "note": ""}
                if np.allclose(diffs, diffs[0]):
                    row["note"] = "identical difference in every fold: t-test undefined"
                else:
                    result = paired_f1_test(a, b)
                    row["t_statistic"], row["p_value"] = result["t_statistic"], result["p_value"]
                tests.append(row)
        adjusted = holm_adjust([t["p_value"] for t in tests])
        for t, adj in zip(tests, adjusted):
            t["p_holm"] = adj
            t["significant_p05"] = t["p_value"] is not None and t["p_value"] < alpha
            t["significant_holm"] = adj is not None and adj < alpha
        _write_csv(out / "significance.csv", tests)

    report = {"folds_done": done, "k": k, "alpha": alpha, "n_tests": len(tests),
              "significant_unadjusted": sum(t["significant_p05"] for t in tests),
              "significant_holm": sum(t["significant_holm"] for t in tests),
              "undefined_tests": sum(t["p_value"] is None for t in tests)}
    (out / "cv_report.json").write_text(json.dumps(report, indent=1))
    print(f"aggregated {len(done)} folds ({done}); {report['n_tests']} paired t-tests, "
          f"{report['significant_unadjusted']} significant at p<{alpha} "
          f"({report['significant_holm']} after Holm correction, {report['undefined_tests']} undefined)")
    return report


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="outputs/cv")
    ap.add_argument("--folds", type=int, nargs="+", help="run only these folds (default: all)")
    ap.add_argument("--archs", nargs="+", help="backbones (default: model.architectures from the config)")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--max-distance", type=int, default=5, help="dHash bits for near-duplicate grouping")
    ap.add_argument("--epochs", type=int, help="override train.epochs (for a quick test)")
    ap.add_argument("--batch-size", type=int, default=64, help="evaluation batch size")
    ap.add_argument("--workers", type=int, default=None, help="CPU processes for hashing and watermark scoring")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--skip-provenance", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true", help="only combine the folds already finished")
    args = ap.parse_args()
    cfg = load_config(args.config)
    k = int(cfg["data"]["cv_folds"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    alpha = float(cfg["evaluation"]["significance_alpha"])

    if not args.aggregate_only:
        import torch

        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        archs = args.archs or list(cfg["model"]["architectures"])
        fold_rows = make_folds(cfg, out, k, args.max_distance, args.workers)
        for f in args.folds if args.folds is not None else range(k):
            if not 0 <= f < k:
                ap.error(f"fold {f} is outside 0..{k - 1}")
            fold_dir = out / f"fold_{f}"
            splits = fold_splits(fold_rows, f, args.val_fraction, int(cfg["seed"]), fold_dir)
            print(f"== fold {f}: {len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test", flush=True)
            for i, arch in enumerate(archs):
                train_backbone(cfg, arch, splits, fold_dir / arch, int(cfg["seed"]) + 100 * f + i, device, args.epochs)
            wm = calibrate_watermark(cfg, splits["val"], fold_dir / "watermark.json", args.workers)
            print(f"  watermark: threshold={wm['threshold']:.5f}, P(AI | detected)={wm['confidence']:.4f}", flush=True)
            evaluate_fold(cfg, archs, splits, fold_dir, wm, device, args.batch_size, not args.skip_provenance)
    aggregate(out, k, alpha)


if __name__ == "__main__":
    main()
