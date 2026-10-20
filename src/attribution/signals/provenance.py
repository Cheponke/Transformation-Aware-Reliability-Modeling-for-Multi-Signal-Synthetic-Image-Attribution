"""C2PA provenance channel (manuscript sections 2.2, 3.2.3).

None of the three source datasets ship with Content Credentials, so the
experiment has to sign images itself, then measure how often the manifest
survives the social-media transforms. Both functions below are placeholders
until the signing setup (c2pa-python + a test certificate) is chosen.
"""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ProvenanceStatus(Enum):
    VALID = "valid"        # manifest present, signature and hash binding verify
    INVALID = "invalid"    # manifest present but tampered or unverifiable
    ABSENT = "absent"      # no manifest (e.g. stripped on upload)


@dataclass(frozen=True)
class ProvenanceResult:
    status: ProvenanceStatus
    claims_ai: bool | None = None  # from the manifest's digitalSourceType, when present


def sign(path: str | Path, out_path: str | Path, claims_ai: bool) -> None:
    raise NotImplementedError("C2PA signing not set up yet")


def verify(path: str | Path) -> ProvenanceResult:
    raise NotImplementedError("C2PA verification not set up yet")
