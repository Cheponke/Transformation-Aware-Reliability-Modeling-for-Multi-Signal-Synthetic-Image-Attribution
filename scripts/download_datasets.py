"""Download the datasets from Kaggle into data/raw/<name>/.

Needs the Kaggle CLI (`pip install kaggle`) and an API token at
~/.kaggle/kaggle.json (Kaggle > Settings > Create New Token).
"""
import argparse
import subprocess
from pathlib import Path

from attribution.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--datasets", nargs="+", help="dataset names from the config (default: all)")
    args = ap.parse_args()
    datasets = load_config(args.config)["data"]["datasets"]
    for name in args.datasets or list(datasets):
        spec = datasets[name]
        out = Path(spec["path"])
        if out.exists() and any(out.iterdir()):
            print(f"{name}: already present in {out}, skipping")
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"{name}: downloading {spec['kaggle']} -> {out}")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", spec["kaggle"], "-p", str(out), "--unzip"],
            check=True,
        )


if __name__ == "__main__":
    main()
