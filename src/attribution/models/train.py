"""Training loop (manuscript section 3.6.2): Adam, BCE loss, weight decay, best-on-val checkpoint."""
from pathlib import Path

import torch
import torch.nn as nn


def train(model: nn.Module, train_loader, val_loader, cfg: dict, out_dir: str | Path, device: str = "cpu") -> Path:
    t = cfg["train"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss, best_path = float("inf"), out_dir / "best.pt"
    for epoch in range(t["epochs"]):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)).squeeze(1), y.to(device))
            loss.backward()
            opt.step()
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        print(f"epoch {epoch + 1}: val_loss={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)
    return best_path


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        total += loss_fn(model(x.to(device)).squeeze(1), y.to(device)).item() * len(y)
        n += len(y)
    return total / max(n, 1)
