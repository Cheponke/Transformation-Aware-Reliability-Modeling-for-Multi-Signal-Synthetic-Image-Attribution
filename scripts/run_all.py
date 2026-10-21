"""Run the whole pipeline in order: download, inspect, splits, harmonize, EDA, train, stress test.

Steps whose output already exists are skipped, so re-running after a crash or
disconnect picks up where it stopped; once a step re-runs, every step after it
does too. `--from <step>` re-runs that step and everything after it; `--force`
re-runs everything (downloads stay skipped once complete, see download_datasets.py).

    python scripts/run_all.py
    python scripts/run_all.py --arch resnet50
    python scripts/run_all.py --from train
"""
import argparse
import subprocess
import sys
from pathlib import Path

from attribution.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--from", dest="start", help="re-run from this step onward")
    ap.add_argument("--force", action="store_true", help="re-run every step")
    args = ap.parse_args()
    d = load_config(args.config)["data"]
    processed, harmonized = Path(d["processed_dir"]), Path(d["harmonized_dir"])
    cfg = ["--config", args.config]

    # (name, script arguments, file that exists once the step has finished)
    steps = [
        ("download", ["download_datasets.py", *cfg], None),
        ("inspect", ["inspect_datasets.py", *cfg], None),
        ("prepare", ["prepare_data.py", *cfg], processed / "test.csv"),
        ("harmonize", ["harmonize.py", *cfg], harmonized / "test.csv"),
        ("eda", ["eda.py", *cfg], Path("outputs/eda/report.json")),
        ("train", ["train.py", *cfg, "--arch", args.arch], Path(f"outputs/models/{args.arch}/calibration.json")),
        ("stress_test", ["stress_test.py", *cfg, "--arch", args.arch],
         Path(f"outputs/stress_test/{args.arch}/results.csv")),
    ]
    names = [s[0] for s in steps]
    if args.start and args.start not in names:
        ap.error(f"--from must be one of: {', '.join(names)}")
    rerun_from = names.index(args.start) if args.start else len(names)

    scripts = Path(__file__).parent
    for i, (name, cmd, done) in enumerate(steps):
        if not args.force and i < rerun_from and done is not None and done.exists():
            print(f"== {name}: already done ({done}), skipping", flush=True)
            continue
        print(f"== {name}", flush=True)
        subprocess.run([sys.executable, str(scripts / cmd[0]), *cmd[1:]], check=True)
        if done is not None:  # later steps read this one's output, so they re-run too
            rerun_from = min(rerun_from, i)
    print(f"== all steps finished; results in outputs/stress_test/{args.arch}/results.csv")


if __name__ == "__main__":
    main()
