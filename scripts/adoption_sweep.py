"""Watermark-adoption sweep: how do the systems compare when only some AI images carry a watermark?

The main evaluation assumes every AI image is watermarked (and every image is
signed). Real generators don't all watermark, so this recombines two saved
runs of the same model on the same test images:

- outputs/evaluation/<arch>/per_image.csv            (every AI image watermarked)
- outputs/evaluation/<arch>_no_watermark/per_image.csv (no image watermarked)

For an adoption rate p, each AI image is watermarked with probability p: its
CNN probability and watermark score come from the watermarked run if it is,
from the unmarked run if not. Human images are never watermarked, so they come
from the watermarked run either way. The watermark detector is re-applied to
every image (its threshold is fixed; an unmarked AI image is detected only as
often as a human one), and the fusion rule is re-run. Optionally the C2PA
credential is also present on only a share of images (--provenance-adoption).

Each (p, provenance adoption) is repeated with --repeats random draws of which
images are marked, and the mean and standard deviation over draws are
reported. No model or GPU is needed.

Writes outputs/evaluation/adoption/<arch>/adoption_by_draw.csv, adoption_summary.csv
and crossover.csv (the lowest adoption at which multi-signal F1 is at least
watermark-only F1, per condition).
Usage: python scripts/adoption_sweep.py --arch efficientnet_b0
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attribution.config import load_config  # noqa: E402
from attribution.fusion.decision import PROVENANCE_CONFIDENCE, Evidence, decide  # noqa: E402
from attribution.signals.provenance import ProvenanceResult, ProvenanceStatus  # noqa: E402

DEFAULT_ADOPTION = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
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


def _claims(value: str):
    return {"True": True, "False": False}.get(value)


def combine(marked_row: dict, unmarked_row: dict, is_marked: bool, is_signed: bool,
            band: tuple[float, float], watermark_confidence: float) -> dict:
    """One per-image row as it would be if this AI image was (not) watermarked and (not) signed."""
    src = marked_row if is_marked else unmarked_row
    y = int(marked_row["y_true"])
    prob_ai = float(src["cnn_prob_ai"])
    threshold = float(marked_row["watermark_threshold"])
    wm_detected = float(src["watermark_score"]) >= threshold
    status = ProvenanceStatus(marked_row["provenance_status"]) if is_signed else ProvenanceStatus.ABSENT
    prov = ProvenanceResult(status, _claims(marked_row["provenance_claims_ai"]) if is_signed else None)
    fused = decide(Evidence(provenance=prov, watermark_detected=wm_detected, cnn_prob_ai=prob_ai),
                   abstain_low=band[0], abstain_high=band[1], watermark_confidence=watermark_confidence)
    prov_label = "uncertain"
    if prov.status is ProvenanceStatus.VALID and prov.claims_ai is not None:
        prov_label = "ai" if prov.claims_ai else "human"
    return {
        "condition": marked_row["condition"],
        "y_true": y,
        "cnn_prob_ai": prob_ai,
        "provenance_prediction": prov_label,
        "watermark_prediction": "ai" if wm_detected else "uncertain",
        "cnn_prediction": "ai" if prob_ai >= 0.5 else "human",
        "fusion_prediction": fused.label,
        "fusion_source": fused.source,
        "provenance_confidence": PROVENANCE_CONFIDENCE if prov_label != "uncertain" else None,
        "watermark_confidence": watermark_confidence if wm_detected else None,
        "cnn_confidence": max(prob_ai, 1 - prob_ai),
        "fusion_confidence": float(fused.confidence),
    }


def load_pairs(eval_dir: Path, arch: str) -> tuple[list[tuple[dict, dict]], float]:
    """Matched (watermarked-run row, unmarked-run row) per image and condition, plus the watermark confidence."""
    marked = _read(eval_dir / arch / "per_image.csv")
    unmarked = _read(eval_dir / f"{arch}_no_watermark" / "per_image.csv")
    index = {(r["condition"], os.path.normpath(r["path"])): r for r in unmarked}
    pairs = []
    for r in marked:
        other = index.get((r["condition"], os.path.normpath(r["path"])))
        if other is None:
            raise RuntimeError(f"{r['path']} ({r['condition']}) is missing from the no-watermark run: "
                               "both runs must use the same test split")
        pairs.append((r, other))
    meta = json.loads((eval_dir / arch / "summary.json").read_text())
    return pairs, float(meta["watermark_confidence"]["confidence"])


def sweep(pairs, watermark_confidence: float, band, adoption: list[float], provenance_adoption: list[float],
          repeats: int, seed: int, ece_bins: int) -> list[dict]:
    import evaluate as ev

    paths = sorted({os.path.normpath(m["path"]) for m, _ in pairs})
    ai_paths = sorted({os.path.normpath(m["path"]) for m, _ in pairs if int(m["y_true"]) == 1})
    out = []
    for p_prov in provenance_adoption:
        for p in adoption:
            n_draws = 1 if p in (0.0, 1.0) and p_prov in (0.0, 1.0) else repeats
            for draw in range(n_draws):
                rng = np.random.default_rng([seed, int(p * 1000), int(p_prov * 1000), draw])
                marked = {path for path, u in zip(ai_paths, rng.random(len(ai_paths))) if u < p}
                signed = {path for path, u in zip(paths, rng.random(len(paths))) if u < p_prov}
                rows = [combine(m, u, os.path.normpath(m["path"]) in marked, os.path.normpath(m["path"]) in signed,
                                band, watermark_confidence) for m, u in pairs]
                for item in ev.summarize(rows, ece_bins):
                    out.append({"watermark_adoption": p, "provenance_adoption": p_prov, "draw": draw,
                                "condition": item["condition"], "system": item["system"],
                                **{m: item[m] for m in METRICS}})
    return out


def _f1(value) -> float:
    """A system that flags no AI image has F1 = 0 (as in the cross-validation t-tests)."""
    return 0.0 if value in (None, "") else float(value)


def summarize_draws(by_draw: list[dict]) -> tuple[list[dict], list[dict]]:
    keys = list(dict.fromkeys((r["watermark_adoption"], r["provenance_adoption"], r["condition"], r["system"])
                              for r in by_draw))
    summary = []
    for key in keys:
        rows = [r for r in by_draw if (r["watermark_adoption"], r["provenance_adoption"], r["condition"], r["system"]) == key]
        item = dict(zip(["watermark_adoption", "provenance_adoption", "condition", "system"], key))
        item["n_draws"] = len(rows)
        for m in METRICS:
            values = [(_f1(r[m]) if m == "f1" else float(r[m])) for r in rows
                      if m == "f1" or r[m] not in (None, "")]
            item[f"{m}_mean"] = float(np.mean(values)) if values else None
            item[f"{m}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
        summary.append(item)

    crossover = []
    groups = dict.fromkeys((r["provenance_adoption"], r["condition"]) for r in summary)
    for p_prov, condition in groups:
        f1 = {(r["watermark_adoption"], r["system"]): r["f1_mean"] for r in summary
              if r["provenance_adoption"] == p_prov and r["condition"] == condition}
        levels = sorted({a for a, _ in f1})
        fusion_wins = [a for a in levels if f1[(a, "multi_signal")] >= f1[(a, "watermark_only")]]
        crossover.append({
            "provenance_adoption": p_prov,
            "condition": condition,
            "fusion_at_least_watermark_from": min(fusion_wins) if fusion_wins else None,
            "fusion_at_least_watermark_up_to": max(fusion_wins) if fusion_wins else None,
            "levels_where_fusion_at_least_watermark": ";".join(f"{a:g}" for a in fusion_wins),
            **{f"f1_fusion_at_{a:g}": f1[(a, "multi_signal")] for a in levels},
            **{f"f1_watermark_at_{a:g}": f1[(a, "watermark_only")] for a in levels},
            **{f"f1_cnn_at_{a:g}": f1[(a, "cnn_only")] for a in levels},
        })
    return summary, crossover


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--eval-dir", default="outputs/evaluation")
    ap.add_argument("--out", default="outputs/evaluation/adoption")
    ap.add_argument("--adoption", type=float, nargs="+", default=DEFAULT_ADOPTION, help="watermark adoption rates")
    ap.add_argument("--provenance-adoption", type=float, nargs="+", default=[1.0], help="C2PA adoption rates")
    ap.add_argument("--repeats", type=int, default=20, help="random draws per adoption rate")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for p in args.adoption + args.provenance_adoption:
        if not 0 <= p <= 1:
            ap.error("adoption rates must be between 0 and 1")
    band = (float(cfg["fusion"]["abstain_low"]), float(cfg["fusion"]["abstain_high"]))

    pairs, confidence = load_pairs(Path(args.eval_dir), args.arch)
    n_draws = sum(1 if p in (0.0, 1.0) and q in (0.0, 1.0) else args.repeats
                  for p in args.adoption for q in args.provenance_adoption)
    print(f"{args.arch}: {len(pairs)} image-condition pairs, {n_draws} draws", flush=True)
    by_draw = sweep(pairs, confidence, band, sorted(args.adoption), sorted(args.provenance_adoption), args.repeats,
                    int(cfg["seed"]), int(cfg["evaluation"]["ece_bins"]))
    summary, crossover = summarize_draws(by_draw)
    out = Path(args.out) / args.arch
    _write(out / "adoption_by_draw.csv", by_draw)
    _write(out / "adoption_summary.csv", summary)
    _write(out / "crossover.csv", crossover)
    for row in crossover:
        print(f"  provenance {row['provenance_adoption']:g} | {row['condition']:>14}: fusion F1 >= watermark-only F1 at "
              f"adoption {row['levels_where_fusion_at_least_watermark'] or 'none'}", flush=True)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
