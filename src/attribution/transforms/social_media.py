"""Simulated social-media processing (manuscript section 3.4.2).

Every transform re-encodes the image, which also drops embedded metadata,
mirroring how platforms strip EXIF/C2PA data on upload.
"""
import io
from typing import Callable

from PIL import Image

Transform = Callable[[Image.Image], Image.Image]


def encode(img: Image.Image, fmt: str, quality: int) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=quality)
    return buf.getvalue()


def _encode(img: Image.Image, fmt: str, quality: int) -> Image.Image:
    out = Image.open(io.BytesIO(encode(img, fmt, quality)))
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


def delivery_format(condition: str, cfg: dict) -> tuple[str, int]:
    """File format and quality a platform would serve after `condition`, for checks on the file itself (C2PA).

    Resized images are served as JPEG at the harmonized quality; other conditions
    keep the encoding they apply (re-encoding at the same quality changes little).
    """
    t = cfg["transforms"]
    if condition.startswith("jpeg_q"):
        return "JPEG", int(condition[6:])
    if condition.startswith("webp_q"):
        return "WEBP", int(condition[6:])
    if condition.startswith("recompress_x"):
        return "JPEG", t["recompression_quality"]
    return "JPEG", cfg["preprocess"]["harmonize"]["jpeg_quality"]
