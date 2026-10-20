"""PyTorch dataset with optional social-media transformation applied before tensor preprocessing."""
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from attribution.data.datasets import Record


def train_transform(cfg: dict) -> T.Compose:
    p, a = cfg["preprocess"], cfg["augment"]
    ops = [T.Resize((p["image_size"], p["image_size"])), T.RandomRotation(a["rotation_deg"])]
    if a["hflip"]:
        ops.append(T.RandomHorizontalFlip())
    jitter = 0.1 if a["color_jitter"] else 0.0
    ops.append(T.ColorJitter(a["brightness"], a["contrast"], saturation=jitter, hue=jitter / 2))
    ops += [T.ToTensor(), T.Normalize(p["mean"], p["std"])]
    return T.Compose(ops)


def eval_transform(cfg: dict) -> T.Compose:
    p = cfg["preprocess"]
    return T.Compose([
        T.Resize((p["image_size"], p["image_size"])),
        T.ToTensor(),
        T.Normalize(p["mean"], p["std"]),
    ])


class ImageRecords(Dataset):
    def __init__(
        self,
        records: list[Record],
        tensor_transform: Callable,
        degradation: Callable[[Image.Image], Image.Image] | None = None,
    ):
        self.records = records
        self.tensor_transform = tensor_transform
        self.degradation = degradation

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        img = Image.open(r.path).convert("RGB")
        if self.degradation is not None:
            img = self.degradation(img)
        return self.tensor_transform(img), torch.tensor(r.label, dtype=torch.float32)
