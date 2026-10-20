"""Reactive CNN channel: calibrated probability that an image is AI-generated."""
import torch

from attribution.models.calibration import sigmoid


@torch.no_grad()
def predict_logits(model: torch.nn.Module, loader, device: str = "cpu") -> tuple[list[float], list[int]]:
    model.eval().to(device)
    logits, labels = [], []
    for x, y in loader:
        logits.extend(model(x.to(device)).squeeze(1).cpu().tolist())
        labels.extend(y.int().tolist())
    return logits, labels


def calibrated_probs(logits, temperature: float):
    return sigmoid([z / temperature for z in logits])
