import io

from PIL import Image

import numpy as np
import pytest

from attribution.signals import provenance as prov
from attribution.signals.provenance import ProvenanceStatus
from attribution.transforms.social_media import downscale, encode, jpeg


def test_phash_stable_under_compression_but_not_across_images(art_image):
    h = np.array([prov.phash(art_image), prov.phash(jpeg(50)(art_image)), prov.phash(downscale(0.5)(art_image))], dtype=np.uint64)
    other = np.array([prov.phash(art_image.transpose(0))], dtype=np.uint64)  # mirrored
    assert prov.hamming(h[:1], h).max() <= 4
    assert prov.hamming(h[:1], other)[0, 0] > 16


@pytest.fixture(scope="module")
def setup():
    pytest.importorskip("c2pa")
    creds = prov.make_credentials()
    return creds, prov.signer(creds), prov.context([creds.root_pem])


def _signed(art_image, setup, source_type="trainedAlgorithmicMedia"):
    _, signer, ctx = setup
    return prov.sign(encode(art_image, "JPEG", 95), "image/jpeg", source_type, signer, ctx)


def test_signed_image_verifies_with_its_claim(art_image, setup):
    signed, _ = _signed(art_image, setup)
    assert prov.verify(signed, "image/jpeg", setup[2]) == prov.ProvenanceResult(ProvenanceStatus.VALID, True)
    human, _ = _signed(art_image, setup, "digitalCreation")
    assert prov.verify(human, "image/jpeg", setup[2]).claims_ai is False


def test_reencode_strips_and_copied_manifest_is_invalid(art_image, setup):
    signed, store = _signed(art_image, setup)
    segments = prov.jpeg_manifest_segments(signed)
    with Image.open(io.BytesIO(signed)) as f:
        img = f.convert("RGB")
    for fmt, mime in [("JPEG", "image/jpeg"), ("WEBP", "image/webp")]:
        out = encode(img, fmt, 90)
        assert prov.verify(out, mime, setup[2]).status is ProvenanceStatus.ABSENT
        kept = prov.carry_manifest(out, mime, segments, store)
        assert prov.verify(kept, mime, setup[2]).status is ProvenanceStatus.INVALID


def test_carry_manifest_onto_identical_bytes_stays_valid(art_image, setup):
    # sanity check of the splicing itself: the original bytes plus their own manifest verify again
    signed, store = _signed(art_image, setup)
    src = encode(art_image, "JPEG", 95)
    assert prov.verify(prov.carry_manifest(src, "image/jpeg", prov.jpeg_manifest_segments(signed), store),
                       "image/jpeg", setup[2]).status is ProvenanceStatus.VALID


def test_untrusted_signer(art_image, setup):
    forger = prov.make_credentials("Forger")
    forged, _ = prov.sign(encode(art_image, "JPEG", 95), "image/jpeg", "digitalCreation", prov.signer(forger), setup[2])
    assert prov.verify(forged, "image/jpeg", setup[2]).status is ProvenanceStatus.UNTRUSTED
