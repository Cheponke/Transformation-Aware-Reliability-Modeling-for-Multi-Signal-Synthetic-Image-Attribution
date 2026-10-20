"""Index raw dataset folders into (path, label, source) records.

Each public dataset ships with its own folder layout. CLASS_DIRS maps the
folder names inside each dataset root to our labels (0 = human, 1 = AI).
Adjust the folder names once the datasets are downloaded.
"""
from dataclasses import dataclass
from pathlib import Path

HUMAN, AI = 0, 1
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CLASS_DIRS = {
    "kannan_ai_human_art": {"human": HUMAN, "ai": AI},
    "superpotato9_ai_recognition": {"real": HUMAN, "ai": AI},
    "ai_artbench": {"human": HUMAN, "ai": AI},
}


@dataclass(frozen=True)
class Record:
    path: str
    label: int
    source: str


def index_dataset(name: str, root: str | Path) -> list[Record]:
    root = Path(root)
    records = []
    for folder, label in CLASS_DIRS[name].items():
        class_root = root / folder
        if not class_root.exists():
            raise FileNotFoundError(f"{name}: expected class folder {class_root}")
        for p in sorted(class_root.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                records.append(Record(str(p), label, name))
    return records
