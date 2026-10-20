from attribution.config import load_config
from attribution.transforms.social_media import build_conditions, downscale, jpeg


def test_conditions_cover_manuscript_settings():
    names = build_conditions(load_config())
    for q in (90, 70, 50, 30):
        assert f"jpeg_q{q}" in names
    assert "clean" in names and names["clean"] is None


def test_transforms_return_rgb_and_drop_exif(art_image):
    out = jpeg(50)(art_image)
    assert out.mode == "RGB" and out.size == art_image.size
    assert not out.getexif()
    assert downscale(0.5)(art_image).size == (128, 128)
