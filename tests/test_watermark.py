from attribution.signals import watermark
from attribution.transforms.social_media import jpeg

KEY, ALPHA, THRESHOLD = 1234, 4.0, 0.05


def test_detects_embedded_mark_and_not_clean(art_image):
    marked = watermark.embed(art_image, ALPHA, KEY)
    assert watermark.detect(marked, KEY, THRESHOLD)[0]
    assert not watermark.detect(art_image, KEY, THRESHOLD)[0]


def test_wrong_key_not_detected(art_image):
    marked = watermark.embed(art_image, ALPHA, KEY)
    assert not watermark.detect(marked, KEY + 1, THRESHOLD)[0]


def test_score_drops_under_compression(art_image):
    marked = watermark.embed(art_image, ALPHA, KEY)
    assert watermark.score(jpeg(30)(marked), KEY) < watermark.score(marked, KEY)


def test_psnr_matches_strength(art_image):
    # a +/-alpha pattern gives MSE alpha^2, so PSNR = 20 log10(255 / alpha) where nothing clips
    assert abs(watermark.psnr(art_image, watermark.embed(art_image, ALPHA, KEY)) - 36.1) < 0.5


def test_survives_moderate_jpeg(art_image):
    marked = watermark.embed(art_image, ALPHA, KEY)
    assert watermark.detect(jpeg(50)(marked), KEY, THRESHOLD)[0]
    assert not watermark.detect(jpeg(50)(art_image), KEY, THRESHOLD)[0]
