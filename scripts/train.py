"""Train one backbone on the prepared splits, then fit a temperature on the validation set."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from attribution.config import load_config, splits_dir
from attribution.data.splits import read_split
from attribution.data.torch_dataset import ImageRecords, eval_transform, train_transform
from attribution.models.calibration import fit_temperature
from attribution.models.cnn import build_model
from attribution.models.train import train
from attribution.signals.forensic import predict_logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--out", default="outputs/models")
    ap.add_argument("--epochs", type=int, help="override train.epochs from the config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed = splits_dir(cfg)
    bs, workers = cfg["train"]["batch_size"], cfg["train"]["num_workers"]

    train_ds = ImageRecords(read_split(processed / "train.csv"), train_transform(cfg))
    val_ds = ImageRecords(read_split(processed / "val.csv"), eval_transform(cfg))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=bs, num_workers=workers)

    model = build_model(args.arch, cfg["model"]["pretrained"], cfg["model"]["dropout"])
    out_dir = Path(args.out) / args.arch
    print(f"training {args.arch} on {device}: {len(train_ds)} train / {len(val_ds)} val images")
    best = train(model, train_loader, val_loader, cfg, out_dir, device, args.epochs)

    model.load_state_dict(torch.load(best, map_location=device))
    logits, labels = predict_logits(model, val_loader, device)
    temperature = fit_temperature(logits, labels)
    (out_dir / "calibration.json").write_text(json.dumps({"temperature": temperature}))
    print(f"saved {best}, temperature={temperature:.3f}")


if __name__ == "__main__":
    main()
