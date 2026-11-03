"""Per-generator report for the external (unseen-generator) test, next to the in-distribution results.

Reads the per-image results that scripts/evaluate.py wrote for the external set
(--split-csv data/external/external.csv --out outputs/evaluation/external),
with and without the watermark, and the in-distribution runs from notebook 05.

For each run, condition and system it reports:
- per generator: metrics on that generator's AI images together with all human
  images (recall = share of that generator's images flagged as AI);
- overall external vs in-distribution: the same metrics on the whole external
  set and on the original test split, and the drop between them.

Writes outputs/evaluation/external/report/{by_generator.csv, vs_in_distribution.csv}.
Usage: python scripts/external_report.py
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate as ev  # noqa: E402
from attribution.config import load_config  # noqa: E402

METRICS = ["coverage", "accuracy", "precision", "recall", "fpr", "f1", "ece"]


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def by_generator(rows: list[dict], ece_bins: int) -> list[dict]:
    """Each generator's AI images plus every human image, summarized per condition and system."""
    human = [r for r in rows if int(r["y_true"]) == 0]
    generators = sorted({r["source"] for r in rows if int(r["y_true"]) == 1})
    out = []
    for g in ["all"] + generators:
        subset = rows if g == "all" else human + [r for r in rows if int(r["y_true"]) == 1 and r["source"] == g]
        for item in ev.summarize(subset, ece_bins):
            out.append({"generator": g, "condition": item["condition"], "system": item["system"],
                        "n_ai": item["n_ai"], "n_human": item["n_human"], **{m: item[m] for m in METRICS}})
    return out


def compare(external: list[dict], internal: list[dict]) -> list[dict]:
    ext = {(r["condition"], r["system"]): r for r in external if r["generator"] == "all"}
    out = []
    for (condition, system), e in ext.items():
        i = internal.get((condition, system))
        if i is None:
            continue
        row = {"condition": condition, "system": system}
        for m in METRICS:
            a = None if i.get(m) in (None, "") else float(i[m])
            b = e[m]
            row[f"{m}_in_distribution"] = a
            row[f"{m}_external"] = b
            row[f"{m}_change"] = (b - a) if a is not None and b is not None else None
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--external", default="outputs/evaluation/external")
    ap.add_argument("--internal", default="outputs/evaluation")
    ap.add_argument("--archs", nargs="+", default=["resnet50", "efficientnet_b0"])
    args = ap.parse_args()
    ece_bins = int(load_config(args.config)["evaluation"]["ece_bins"])
    ext_root, int_root = Path(args.external), Path(args.internal)

    all_by_gen, all_cmp = [], []
    for arch in args.archs:
        for run in (arch, f"{arch}_no_watermark"):
            path = ext_root / run / "per_image.csv"
            if not path.is_file():
                print(f"skipping {run}: {path} not found")
                continue
            gen = by_generator(_read(path), ece_bins)
            all_by_gen += [{"run": run, **r} for r in gen]
            internal_summary = int_root / run / "summary.csv"
            if internal_summary.is_file():
                internal = {(r["condition"], r["system"]): r for r in _read(internal_summary)}
                all_cmp += [{"run": run, **r} for r in compare(gen, internal)]
            fusion = {r["condition"]: r for r in gen if r["generator"] == "all" and r["system"] == "multi_signal"}
            cnn = {r["condition"]: r for r in gen if r["generator"] == "all" and r["system"] == "cnn_only"}
            print(f"{run}: external clean accuracy CNN {cnn['clean']['accuracy']:.3f}, fusion {fusion['clean']['accuracy']:.3f}")
    if not all_by_gen:
        raise RuntimeError(f"no external results in {ext_root}: run evaluate.py with --split-csv first")
    out = ext_root / "report"
    _write(out / "by_generator.csv", all_by_gen)
    if all_cmp:
        _write(out / "vs_in_distribution.csv", all_cmp)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
