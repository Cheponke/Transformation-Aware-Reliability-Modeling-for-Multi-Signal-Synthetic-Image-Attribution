import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def art_image():
    """Smooth synthetic 'painting' so compression behaves like it does on real images."""
    h = w = 256
    yy, xx = np.mgrid[0:h, 0:w]
    r = 128 + 80 * np.sin(xx / 17.0)
    g = 128 + 80 * np.cos(yy / 23.0)
    b = 128 + 60 * np.sin((xx + yy) / 31.0)
    return Image.fromarray(np.stack([r, g, b], -1).clip(0, 255).astype(np.uint8))
