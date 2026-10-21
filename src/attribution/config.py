from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def splits_dir(cfg: dict) -> Path:
    """Where training and evaluation read the split CSVs from."""
    d = cfg["data"]
    return Path(d["harmonized_dir"] if cfg["preprocess"]["harmonize"]["enabled"] else d["processed_dir"])
