"""C2PA provenance channel (manuscript sections 2.2, 3.2.3).

None of the three source datasets ship with Content Credentials, so the
experiment signs images itself with a test certificate authority (as a
generator or a creative tool would), then checks what a verifier sees after
social-media processing. Verification follows the C2PA rules via c2pa-python:
the manifest must be present, its signature valid, its hard binding (a hash
over the image bytes) must still match, and the signer must chain to a trusted
root. Any re-encode changes the bytes, so a copied manifest becomes invalid.

C2PA also allows soft binding: a perceptual fingerprint of the content, looked
up in a manifest repository, re-associates an image with its credential after
the embedded manifest is stripped (section 2.2). `phash` is that fingerprint.
"""
import datetime
import io
import struct
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image

AI_SOURCE_TYPES = {"trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia", "algorithmicMedia", "compositeSynthetic"}
IPTC = "http://cv.iptc.org/newscodes/digitalsourcetype/"


class ProvenanceStatus(Enum):
    VALID = "valid"            # manifest present, signature and hash binding verify, signer trusted
    UNTRUSTED = "untrusted"    # manifest intact but signed by a certificate not on the trust list
    INVALID = "invalid"        # manifest present but tampered or no longer bound to these bytes
    ABSENT = "absent"          # no manifest (e.g. stripped on upload)


@dataclass(frozen=True)
class ProvenanceResult:
    status: ProvenanceStatus
    claims_ai: bool | None = None  # from the manifest's digitalSourceType, when present


@dataclass(frozen=True)
class Credentials:
    """PEM strings for a test root CA and a signing certificate it issued."""
    root_pem: str
    chain_pem: str   # signing cert followed by the root
    key_pem: str


def make_credentials(org: str = "Thesis test CA") -> Credentials:
    """ES256 root CA + end-entity signing certificate with the extensions C2PA requires."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    def name(cn):
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn), x509.NameAttribute(NameOID.ORGANIZATION_NAME, org)])

    def usage(sign: bool):
        return x509.KeyUsage(digital_signature=sign, content_commitment=False, key_encipherment=False, data_encipherment=False,
                             key_agreement=False, key_cert_sign=not sign, crl_sign=not sign, encipher_only=False, decipher_only=False)

    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key, key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    root = (x509.CertificateBuilder().subject_name(name("Root")).issuer_name(name("Root"))
            .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(usage(False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    leaf = (x509.CertificateBuilder().subject_name(name("Signer")).issuer_name(root.subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(usage(True), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    pem = lambda c: c.public_bytes(serialization.Encoding.PEM).decode()
    key_pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    return Credentials(pem(root), pem(leaf) + pem(root), key_pem)


def signer(creds: Credentials):
    import c2pa
    return c2pa.Signer.from_info(c2pa.C2paSignerInfo(c2pa.C2paSigningAlg.ES256, creds.chain_pem.encode(), creds.key_pem.encode(), None))


def context(trusted_roots: list[str]):
    """Verification settings: which roots are trusted; no thumbnail, so signed files stay small."""
    import c2pa
    return c2pa.Context.from_dict({"trust": {"trust_anchors": "".join(trusted_roots)},
                                   "builder": {"thumbnail": {"enabled": False}}})


def sign(data: bytes, mime: str, source_type: str, signer_, ctx=None) -> tuple[bytes, bytes]:
    """Embed a signed manifest declaring how the image was made (an IPTC digitalSourceType).

    Returns the signed file and the raw manifest store.
    """
    import c2pa
    manifest = {"claim_generator_info": [{"name": "thesis-attribution", "version": "0.1"}],
                "assertions": [{"label": "c2pa.actions", "data": {"actions": [
                    {"action": "c2pa.created", "digitalSourceType": IPTC + source_type}]}}]}
    out = io.BytesIO()
    with (c2pa.Builder(manifest, context=ctx) if ctx is not None else c2pa.Builder(manifest)) as b:
        store = b.sign(signer_, mime, io.BytesIO(data), out)
    return out.getvalue(), bytes(store)


def verify(data: bytes, mime: str, ctx) -> ProvenanceResult:
    import c2pa
    try:
        with c2pa.Reader(mime, io.BytesIO(data), context=ctx) as r:
            state = r.get_validation_state()
            manifest = r.get_active_manifest() or {}
    except c2pa.C2paError as e:
        if "ManifestNotFound" in type(e).__name__ or "no JUMBF" in str(e):
            return ProvenanceResult(ProvenanceStatus.ABSENT)
        return ProvenanceResult(ProvenanceStatus.INVALID)
    claims_ai = _claims_ai(manifest)
    if state == "Trusted":
        return ProvenanceResult(ProvenanceStatus.VALID, claims_ai)
    if state == "Valid":  # everything verifies except the signer's trust
        return ProvenanceResult(ProvenanceStatus.UNTRUSTED, claims_ai)
    return ProvenanceResult(ProvenanceStatus.INVALID, claims_ai)


def _claims_ai(manifest: dict) -> bool | None:
    for a in manifest.get("assertions", []):
        if a.get("label", "").startswith("c2pa.actions"):
            for act in a.get("data", {}).get("actions", []):
                t = act.get("digitalSourceType")
                if t:
                    return t.rsplit("/", 1)[-1] in AI_SOURCE_TYPES
    return None


# Carrying a manifest over to a re-encoded file, as a platform that keeps metadata would.

def jpeg_manifest_segments(data: bytes) -> bytes:
    """The APP11 (JUMBF) segments of a JPEG, where C2PA stores its manifest."""
    i, out = 2, []
    while i + 4 <= len(data) and data[i] == 0xFF and data[i + 1] != 0xDA:
        n = struct.unpack(">H", data[i + 2:i + 4])[0]
        if data[i + 1] == 0xEB:
            out.append(data[i:i + 2 + n])
        i += 2 + n
    return b"".join(out)


def carry_manifest(data: bytes, mime: str, segments: bytes, store: bytes) -> bytes:
    """Insert a manifest into `data` where c2pa itself would put it, without re-signing.

    `segments` are the manifest's JPEG APP11 segments (`jpeg_manifest_segments` of the
    signed file), `store` the raw manifest store returned by `sign`.
    """
    if mime == "image/jpeg":
        i = 4 + struct.unpack(">H", data[4:6])[0] if data[2:4] == b"\xff\xe0" else 2  # after JFIF APP0
        return data[:i] + segments + data[i:]
    if mime == "image/webp":
        body = data[12:] + b"C2PA" + struct.pack("<I", len(store)) + store + (b"\0" if len(store) & 1 else b"")
        return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WEBP" + body
    raise ValueError(mime)


# Soft binding: perceptual hash for manifest-repository lookup.

def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    m = np.cos(np.pi * (2 * np.arange(n)[None, :] + 1) * k / (2 * n))
    m[0] /= np.sqrt(2)
    return m


_DCT32 = _dct_matrix(32)


def phash(img: Image.Image) -> int:
    """64-bit DCT perceptual hash: low-frequency structure, which survives compression and resizing."""
    x = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    low = (_DCT32 @ x @ _DCT32.T)[:8, :8].ravel()
    bits = low > np.median(low[1:])
    return int(np.packbits(bits).view(">u8")[0])


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between two arrays of uint64 hashes, shape (len(a), len(b))."""
    x = np.bitwise_xor(a[:, None].astype(np.uint64), b[None, :].astype(np.uint64))
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int16)
    return np.unpackbits(x.view(np.uint8), axis=-1).reshape(*x.shape, 64).sum(-1).astype(np.int16)
