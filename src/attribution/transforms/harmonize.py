"""Bring every image to the same size and file format before training (see config `harmonize`)."""
from PIL import Image, ImageOps


def harmonize(img: Image.Image, size: int) -> Image.Image:
    """Resize the shorter side to `size` and center-crop to a square RGB image."""
    if img.mode in ("P", "LA", "RGBA"):
        # Flatten transparency onto white rather than letting it turn black.
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    return ImageOps.fit(img.convert("RGB"), (size, size), Image.BICUBIC)
