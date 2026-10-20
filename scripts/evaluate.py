"""Evaluate single-signal and multi-signal attribution across the transformation conditions.

Planned steps (not wired up yet):
  1. For each condition in build_conditions(cfg), degrade the test images.
  2. Run each channel: provenance.verify, watermark.detect, calibrated CNN.
  3. Report per-channel availability/survival rates and metrics.
  4. Run fusion.decide and report selective metrics, coverage, ECE and delta-A vs "clean".
  5. Paired t-test of multi- vs single-signal F1 across the 5 CV folds.
"""
import argparse

from attribution.config import load_config
from attribution.transforms.social_media import build_conditions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for name in build_conditions(cfg):
        print(name)
    raise SystemExit("evaluation pipeline not implemented yet")


if __name__ == "__main__":
    main()
