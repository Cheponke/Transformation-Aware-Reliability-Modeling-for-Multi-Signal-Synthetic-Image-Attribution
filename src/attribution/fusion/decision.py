"""Multi-signal decision logic (manuscript sections 3.2, 3.2.4).

Sequential fallback: valid provenance -> detected watermark -> calibrated CNN.
The CNN abstains ("uncertain") when its probability falls inside the abstain band.
"""
from dataclasses import dataclass
from typing import Literal

from attribution.signals.provenance import ProvenanceResult, ProvenanceStatus

Label = Literal["ai", "human", "uncertain"]


@dataclass(frozen=True)
class Evidence:
    provenance: ProvenanceResult | None = None
    watermark_detected: bool | None = None
    watermark_score: float | None = None
    cnn_prob_ai: float | None = None  # calibrated


@dataclass(frozen=True)
class Decision:
    label: Label
    confidence: float
    source: Literal["provenance", "watermark", "cnn", "none"]


PROVENANCE_CONFIDENCE = 0.99
# Default only. scripts/evaluate.py measures P(AI | watermark detected) on the
# validation split (pooled over every transform condition) and passes it in.
WATERMARK_CONFIDENCE = 0.95


def decide(
    e: Evidence,
    abstain_low: float,
    abstain_high: float,
    watermark_confidence: float = WATERMARK_CONFIDENCE,
) -> Decision:
    p = e.provenance
    if p is not None and p.status is ProvenanceStatus.VALID and p.claims_ai is not None:
        return Decision("ai" if p.claims_ai else "human", PROVENANCE_CONFIDENCE, "provenance")
    # A missing watermark says nothing about human origin, so only a detection is used.
    if e.watermark_detected:
        return Decision("ai", float(watermark_confidence), "watermark")
    if e.cnn_prob_ai is not None:
        q = e.cnn_prob_ai
        if abstain_low < q < abstain_high:
            return Decision("uncertain", max(q, 1 - q), "cnn")
        return Decision("ai" if q >= abstain_high else "human", max(q, 1 - q), "cnn")
    return Decision("uncertain", 0.5, "none")
