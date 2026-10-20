"""Download the datasets from Kaggle into data/raw/<name>/.

Needs the Kaggle CLI (`pip install kaggle`) and an API token at
~/.kaggle/kaggle.json (Kaggle > Settings > API Tokens > Legacy API Credentials).

A dataset counts as present only once its download and unzip finished, marked
by a COMPLETE_MARKER file; a folder without it is wiped and downloaded again.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

from attribution.config import load_config

COMPLETE_MARKER = ".download_complete"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--datasets", nargs="+", help="dataset names from the config (default: all)")
    args = ap.parse_args()
    datasets = load_config(args.config)["data"]["datasets"]
    for name in args.datasets or list(datasets):
        spec = datasets[name]
        out = Path(spec["path"])
        if (out / COMPLETE_MARKER).exists():
            print(f"{name}: already present in {out}, skipping")
            continue
        if out.exists():
            print(f"{name}: removing incomplete download in {out}")
            shutil.rmtree(out)
        out.mkdir(parents=True)
        print(f"{name}: downloading {spec['kaggle']} -> {out}")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", spec["kaggle"], "-p", str(out), "--unzip"],
            check=True,
        )
        (out / COMPLETE_MARKER).touch()


if __name__ == "__main__":
    main()
