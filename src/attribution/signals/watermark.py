"""Additive spread-spectrum watermark, I_w = I + alpha * W (manuscript section 3.6.1).

W is a keyed +/-1 pattern generated at a fixed reference size and resized to
the image, so detection still works after the image has been downscaled.
Detection is normalized correlation between the image luminance and W,
compared against a threshold.
"""
import numpy as np
from PIL import Image

REFERENCE_SIZE = 512


def pattern(key: int, size: tuple[int, int], block: int = 4) -> np.ndarray:
    """Keyed +/-1 pattern of shape (h, w); `block` sets its spatial frequency."""
    rng = np.random.default_rng(key)
    n = REFERENCE_SIZE // block
    coarse = rng.choice([-1.0, 1.0], size=(n, n)).astype(np.float32)
    w, h = size
    resized = Image.fromarray(coarse).resize((w, h), Image.NEAREST)
    return np.asarray(resized, dtype=np.float32)


def embed(img: Image.Image, alpha: float, key: int) -> Image.Image:
    x = np.asarray(img.convert("RGB"), dtype=np.float32)
    w = pattern(key, img.size)[..., None]
    return Image.fromarray(np.clip(x + alpha * w, 0, 255).astype(np.uint8))


def score(img: Image.Image, key: int) -> float:
    """Normalized correlation between zero-mean luminance and the keyed pattern."""
    y = np.asarray(img.convert("L"), dtype=np.float32)
    y = y - y.mean()
    w = pattern(key, img.size)
    w = w - w.mean()
    denom = np.linalg.norm(y) * np.linalg.norm(w)
    return float((y * w).sum() / denom) if denom > 0 else 0.0


def detect(img: Image.Image, key: int, threshold: float) -> tuple[bool, float]:
    s = score(img, key)
    return s >= threshold, s
