from attribution.fusion.decision import Evidence, decide
from attribution.signals.provenance import ProvenanceResult, ProvenanceStatus

BAND = dict(abstain_low=0.35, abstain_high=0.65)


def test_valid_provenance_wins():
    e = Evidence(ProvenanceResult(ProvenanceStatus.VALID, claims_ai=False), True, 0.3, 0.99)
    assert decide(e, **BAND).source == "provenance"
    assert decide(e, **BAND).label == "human"


def test_watermark_used_when_provenance_absent():
    e = Evidence(ProvenanceResult(ProvenanceStatus.ABSENT), True, 0.2, 0.1)
    assert decide(e, **BAND).label == "ai"


def test_cnn_fallback_and_abstention():
    assert decide(Evidence(cnn_prob_ai=0.9), **BAND).label == "ai"
    assert decide(Evidence(cnn_prob_ai=0.1), **BAND).label == "human"
    assert decide(Evidence(cnn_prob_ai=0.5), **BAND).label == "uncertain"
    assert decide(Evidence(), **BAND).source == "none"
