"""Training loop (manuscript section 3.6.2): Adam, BCE loss, weight decay, best-on-val checkpoint."""
import json
from pathlib import Path

import torch
import torch.nn as nn


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: dict,
    out_dir: str | Path,
    device: str = "cpu",
    epochs: int | None = None,
) -> Path:
    """Train with early stopping on validation loss; returns the best checkpoint path."""
    t = cfg["train"]
    epochs = epochs or t["epochs"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    # Mixed precision roughly doubles throughput on Colab GPUs; no effect on CPU.
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_loss, best_path, stale, history = float("inf"), out_dir / "best.pt", 0, []
    for epoch in range(epochs):
        model.train()
        train_total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = loss_fn(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_total += loss.item() * len(y)
            n += len(y)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch + 1, "train_loss": train_total / max(n, 1), "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch + 1}: train_loss={train_total / max(n, 1):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= t["patience"]:
                print(f"early stop: no val improvement for {stale} epochs")
                break
    return best_path


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> tuple[float, float]:
    """Mean loss and accuracy (threshold 0.5) over a loader."""
    model.eval()
    total, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).squeeze(1)
        total += loss_fn(logits, y).item() * len(y)
        correct += ((logits > 0).float() == y).sum().item()
        n += len(y)
    return total / max(n, 1), correct / max(n, 1)
