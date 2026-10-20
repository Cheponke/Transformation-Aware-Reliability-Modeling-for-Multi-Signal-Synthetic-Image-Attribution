"""Print each dataset's folder layout and how the config's class rules label it.

Run this after downloading to check the `classes` rules in the config.
"""
import argparse
from collections import Counter
from pathlib import Path

from attribution.config import load_config
from attribution.data.datasets import IMAGE_EXTS, label_for

LABEL_NAMES = {0: "human", 1: "ai", None: "UNMATCHED"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    for name, spec in load_config(args.config)["data"]["datasets"].items():
        root = Path(spec["path"])
        print(f"\n== {name} ({root})")
        if not root.exists():
            print("   not downloaded")
            continue
        folders = Counter(
            p.parent.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS
        )
        for folder, n in sorted(folders.items()):
            print(f"   {n:>7}  {LABEL_NAMES[label_for(folder, spec['classes'])]:<9}  {folder}")


if __name__ == "__main__":
    main()
