"""Post-hoc temperature scaling for binary logits (Guo et al., 2017)."""
import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def nll(logits, labels, temperature: float) -> float:
    p = np.clip(sigmoid(np.asarray(logits) / temperature), 1e-12, 1 - 1e-12)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(logits, labels, lo: float = 0.05, hi: float = 20.0, iters: int = 100) -> float:
    """Golden-section search for T minimizing validation NLL (NLL is unimodal in log T)."""
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(iters):
        if nll(logits, labels, np.exp(c)) < nll(logits, labels, np.exp(d)):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return float(np.exp((a + b) / 2))
