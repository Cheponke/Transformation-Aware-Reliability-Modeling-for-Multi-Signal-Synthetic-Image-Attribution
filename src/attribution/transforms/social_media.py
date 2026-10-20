"""Simulated social-media processing (manuscript section 3.4.2).

Every transform re-encodes the image, which also drops embedded metadata,
mirroring how platforms strip EXIF/C2PA data on upload.
"""
import io
from typing import Callable

from PIL import Image

Transform = Callable[[Image.Image], Image.Image]


def _encode(img: Image.Image, fmt: str, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=quality)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def jpeg(quality: int) -> Transform:
    return lambda img: _encode(img, "JPEG", quality)


def webp(quality: int) -> Transform:
    return lambda img: _encode(img, "WEBP", quality)


def downscale(scale: float) -> Transform:
    def f(img: Image.Image) -> Image.Image:
        w, h = img.size
        return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    return f


def recompress(cycles: int, quality: int) -> Transform:
    """Repeated download/repost cycles: JPEG re-encoding applied `cycles` times."""
    def f(img: Image.Image) -> Image.Image:
        for _ in range(cycles):
            img = _encode(img, "JPEG", quality)
        return img
    return f


def compose(*fs: Transform) -> Transform:
    def f(img: Image.Image) -> Image.Image:
        for t in fs:
            img = t(img)
        return img
    return f


def build_conditions(cfg: dict) -> dict[str, Transform | None]:
    """Named evaluation conditions; "clean" is the untransformed reference for delta-A."""
    t = cfg["transforms"]
    conditions: dict[str, Transform | None] = {"clean": None}
    for q in t["jpeg_qualities"]:
        conditions[f"jpeg_q{q}"] = jpeg(q)
    for q in t["webp_qualities"]:
        conditions[f"webp_q{q}"] = webp(q)
    for s in t["resize_scales"]:
        conditions[f"resize_{s}"] = downscale(s)
    for c in t["recompression_cycles"]:
        conditions[f"recompress_x{c}"] = recompress(c, t["recompression_quality"])
    return conditions
