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


def test_harmonize_gives_same_size_rgb_and_flattens_transparency(art_image):
    from PIL import Image

    from attribution.transforms.harmonize import harmonize

    wide = art_image.resize((768, 512))
    out = harmonize(wide, 256)
    assert out.size == (256, 256) and out.mode == "RGB"

    clear = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    assert harmonize(clear, 256).getpixel((10, 10)) == (255, 255, 255)
