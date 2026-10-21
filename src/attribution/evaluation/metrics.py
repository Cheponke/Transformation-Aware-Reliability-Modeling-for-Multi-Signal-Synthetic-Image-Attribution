"""Metrics (manuscript section 3.7): accuracy/precision/recall/F1, delta-A, plus calibration and abstention."""
import numpy as np


def confusion(y_true, y_pred) -> dict[str, int]:
    t, p = np.asarray(y_true), np.asarray(y_pred)
    return {
        "tp": int(((t == 1) & (p == 1)).sum()),
        "tn": int(((t == 0) & (p == 0)).sum()),
        "fp": int(((t == 0) & (p == 1)).sum()),
        "fn": int(((t == 1) & (p == 0)).sum()),
    }


def classification_report(y_true, y_pred) -> dict[str, float]:
    c = confusion(y_true, y_pred)
    n = sum(c.values())
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = c["fp"] / (c["fp"] + c["tn"]) if c["fp"] + c["tn"] else 0.0
    return {
        "accuracy": (c["tp"] + c["tn"]) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,  # human art wrongly flagged as AI
        **c,
    }


def delta_accuracy(acc_original: float, acc_transformed: float) -> float:
    """Delta-A = A_original - A_compressed; lower means more robust."""
    return acc_original - acc_transformed


def expected_calibration_error(y_true, prob_ai, n_bins: int = 15) -> float:
    y, p = np.asarray(y_true, dtype=float), np.asarray(prob_ai, dtype=float)
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(float) == y).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def selective_report(y_true, labels: list[str]) -> dict[str, float]:
    """Metrics on the non-abstained subset, plus coverage (share of images that got a decision)."""
    keep = [i for i, l in enumerate(labels) if l != "uncertain"]
    y = [y_true[i] for i in keep]
    pred = [1 if labels[i] == "ai" else 0 for i in keep]
    report = classification_report(y, pred) if keep else {}
    report["coverage"] = len(keep) / len(labels) if labels else 0.0
    return report
