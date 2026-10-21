"""Index raw dataset folders into (path, label, source) records.

Each public dataset ships with its own folder layout, so labels come from the
`classes` rules in the config: a glob matched against the file's parent folder
path (relative to the dataset root). The first matching rule wins; files that
match no rule are skipped and counted.
"""
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

HUMAN, AI = 0, 1
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class Record:
    path: str
    label: int
    source: str


def label_for(rel_dir: str, rules: list[dict]) -> int | None:
    for rule in rules:
        if fnmatch(rel_dir, rule["match"]):
            return int(rule["label"])
    return None


def index_dataset(name: str, spec: dict) -> tuple[list[Record], int]:
    """Return the labeled records and the number of images no rule matched."""
    root = Path(spec["path"])
    if not root.exists():
        raise FileNotFoundError(f"{name}: dataset folder {root} not found")
    records, unmatched = [], 0
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        label = label_for(p.parent.relative_to(root).as_posix(), spec["classes"])
        if label is None:
            unmatched += 1
        else:
            records.append(Record(str(p), label, name))
    return records, unmatched
