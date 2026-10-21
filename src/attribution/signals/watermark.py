"""Additive spread-spectrum watermark, I_w = I + alpha * W (manuscript section 3.6.1).

W is a keyed +/-1 pattern generated at a fixed reference size and resized to
the image, so detection still works after the image has been downscaled.
Detection is normalized correlation between the high-pass filtered image
luminance and W, compared against a threshold. The high-pass step removes the
artwork's own smooth content, which would otherwise drown out a mark that is
small enough to be invisible.
"""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageFilter

REFERENCE_SIZE = 512
HIGHPASS_RADIUS = 2


@lru_cache(maxsize=16)
def _pattern(key: int, size: tuple[int, int], block: int) -> np.ndarray:
    rng = np.random.default_rng(key)
    n = REFERENCE_SIZE // block
    coarse = rng.choice([-1.0, 1.0], size=(n, n)).astype(np.float32)
    resized = np.asarray(Image.fromarray(coarse).resize(size, Image.NEAREST), dtype=np.float32)
    resized.setflags(write=False)
    return resized


def pattern(key: int, size: tuple[int, int], block: int = 4) -> np.ndarray:
    """Keyed +/-1 pattern of shape (h, w); `block` sets its spatial frequency."""
    return _pattern(key, tuple(size), block)


def embed(img: Image.Image, alpha: float, key: int) -> Image.Image:
    x = np.asarray(img.convert("RGB"), dtype=np.float32)
    w = pattern(key, img.size)[..., None]
    return Image.fromarray(np.clip(x + alpha * w, 0, 255).astype(np.uint8))


def psnr(original: Image.Image, marked: Image.Image) -> float:
    """Peak signal-to-noise ratio in dB; above ~40 dB the mark is generally invisible."""
    a = np.asarray(original.convert("RGB"), dtype=np.float64)
    b = np.asarray(marked.convert("RGB"), dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255**2 / mse))


def score(img: Image.Image, key: int) -> float:
    """Normalized correlation between the high-passed luminance and the keyed pattern."""
    gray = img.convert("L")
    y = np.asarray(gray, dtype=np.float32) - np.asarray(gray.filter(ImageFilter.GaussianBlur(HIGHPASS_RADIUS)), dtype=np.float32)
    y = y - y.mean()
    w = pattern(key, img.size)
    w = w - w.mean()
    denom = np.linalg.norm(y) * np.linalg.norm(w)
    return float((y * w).sum() / denom) if denom > 0 else 0.0


def detect(img: Image.Image, key: int, threshold: float) -> tuple[bool, float]:
    s = score(img, key)
    return s >= threshold, s
