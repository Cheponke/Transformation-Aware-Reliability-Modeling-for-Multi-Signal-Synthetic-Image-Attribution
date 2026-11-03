"""External test-set preparation, per-generator report, and Grad-CAM."""
import csv
import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gradcam = _load("gradcam")
prepare = _load("prepare_external")
report = _load("external_report")


# ---------------------------------------------------------------- Grad-CAM

class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.body = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU())
        self.head = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.head(self.body(x).mean(dim=(2, 3)))


def test_gradcam_matches_manual_computation():
    model = _Tiny().eval()
    x = torch.randn(3, 3, 8, 8)
    cams, logits = gradcam.GradCAM(model, "body")(x.clone())
    for i in range(3):
        xi = x[i:i + 1].clone().requires_grad_(True)
        acts = model.body(xi)
        logit = model.head(acts.mean(dim=(2, 3))).squeeze()
        target = logit if logit >= 0 else -logit
        grads, = torch.autograd.grad(target, acts)
        cam = torch.relu((grads.mean(dim=(2, 3), keepdim=True) * acts).sum(1))[0].detach().numpy()
        cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-8)
        assert np.allclose(cams[i], cam, atol=1e-5)
        assert logits[i] == pytest.approx(float(logit.detach()), abs=1e-5)


@pytest.mark.parametrize("arch", ["resnet18", "efficientnet_b0"])
def test_gradcam_runs_on_real_backbones(arch):
    from attribution.models.cnn import build_model

    model = build_model(arch, pretrained=False).eval()
    cams, logits = gradcam.GradCAM(model, gradcam.TARGET_LAYERS[arch])(torch.randn(2, 3, 224, 224))
    assert cams.shape == (2, 224, 224) and logits.shape == (2,)
    assert cams.min() >= 0 and cams.max() <= 1 + 1e-6


def test_iou_of_top_regions():
    a = np.zeros((10, 10)); a[:2] = 1
    assert gradcam.iou(gradcam.top_mask(a), gradcam.top_mask(a)) == 1.0
    b = np.zeros((10, 10)); b[-2:] = 1
    assert gradcam.iou(gradcam.top_mask(a), gradcam.top_mask(b)) == 0.0


# ---------------------------------------------------------------- external set

def _art(seed):
    from PIL import Image

    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:300, 0:400]
    f = r.uniform(8, 40, 3)
    img = np.stack([128 + 90 * np.sin(xx / f[0] + seed), 128 + 90 * np.cos(yy / f[1]), 128 + 70 * np.sin((xx + yy) / f[2])], -1)
    return Image.fromarray((img + r.normal(0, 6, img.shape)).clip(0, 255).astype(np.uint8))


def _png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def vlm(tmp_path):
    """Fake WikiArt_VLM zips (matching file names across zips) and a training pool with one planted duplicate."""
    zips = tmp_path / "zips"
    zips.mkdir()
    names = [f"artist_{i:03d}" for i in range(30)]
    with zipfile.ZipFile(zips / "Original.zip", "w") as z:
        for i, n in enumerate(names):
            z.writestr(f"Original/images/{n}.jpg", _png(_art(i)))
    for k, zname in enumerate(["FLUX.zip", "F-Lite.zip", "Stable-Diffusion.zip"]):
        with zipfile.ZipFile(zips / zname, "w") as z:
            for i, n in enumerate(names):
                z.writestr(f"{zname[:-4]}/images/{n}.png", _png(_art(1000 * (k + 1) + i)))
    # training pool: unrelated images, plus a copy of WikiArt original 0
    pool = tmp_path / "harmonized"
    pool.mkdir()
    rows = []
    for split in ("train", "val", "test"):
        for i in range(4):
            p = pool / f"{split}_{i}.jpg"
            _art(5000 + 10 * i + len(split)).resize((256, 256)).save(p, quality=95)
            rows.append(p)
        if split == "train":  # WikiArt originals 0-5 already in training: 0-2 squeezed to a square, 3-5 center-cropped
            from attribution.transforms.harmonize import harmonize

            for j in range(6):
                p = pool / f"leak_{j}.jpg"
                (_art(j).resize((256, 256)) if j < 3 else harmonize(_art(j), 256)).save(p, quality=95)
                rows.append(p)
        with open(pool / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "source"])
            w.writerows([[str(p), i % 2, "toy"] for i, p in enumerate(rows[-10 if split == "train" else -4:])])
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    cfg["data"]["harmonized_dir"] = str(pool)
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    return tmp_path


def test_prepare_external_pairs_harmonizes_and_removes_leaks(vlm):
    out = vlm / "external"
    run = [sys.executable, str(SCRIPTS / "prepare_external.py"), "--config", str(vlm / "cfg.yaml"),
           "--zips", str(vlm / "zips"), "--out", str(out), "--per-generator", "8", "--workers", "1"]
    subprocess.run(run, check=True, capture_output=True, text=True)
    summary = json.loads((out / "external_summary.json").read_text())
    assert summary["sampling"] == "paired"
    rows = list(csv.DictReader((out / "external.csv").open()))
    removed = list(csv.DictReader((out / "removed_near_duplicates.csv").open()))
    ai = [r for r in rows if r["label"] == "1"]
    human_stems = {Path(r["member"]).stem for r in rows + removed if r["label"] == "0"}
    ai_stems = [Path(r["member"]).stem for r in ai + [r for r in removed if r["label"] == "1"]]
    assert len(ai_stems) == len(set(ai_stems)) == 24           # each painting used by one generator only
    assert set(ai_stems) == human_stems                         # ... and its original is the human counterpart
    from PIL import Image

    with Image.open(rows[0]["path"]) as img:
        assert img.size == (256, 256) and img.format == "JPEG"
    leaked = {f"artist_{j:03d}" for j in range(6)} & human_stems  # planted training copies that were sampled
    removed_human = {Path(r["member"]).stem for r in removed if r["label"] == "0"}
    assert leaked and leaked <= removed_human                     # all caught, cropped or squeezed
    assert not (out / "_squeezed").exists()
    assert summary["kept_total"] == len(rows)
    inspect = subprocess.run(run[:6] + ["--inspect"], check=True, capture_output=True, text=True).stdout
    assert "30 of 30 file names also appear" in inspect


def test_by_generator_recall():
    rows = []
    for i in range(10):
        rows.append({"condition": "clean", "y_true": 0, "source": "wikiart", "cnn_prob_ai": 0.1,
                     "provenance_prediction": "uncertain", "watermark_prediction": "uncertain", "cnn_prediction": "human",
                     "fusion_prediction": "human", "fusion_source": "cnn", "provenance_confidence": None,
                     "watermark_confidence": None, "cnn_confidence": 0.9, "fusion_confidence": 0.9})
    for g, hits in [("flux", 4), ("f_lite", 1)]:
        for i in range(5):
            p = 0.9 if i < hits else 0.2
            rows.append({"condition": "clean", "y_true": 1, "source": g, "cnn_prob_ai": p,
                         "provenance_prediction": "uncertain", "watermark_prediction": "uncertain",
                         "cnn_prediction": "ai" if p >= 0.5 else "human", "fusion_prediction": "ai" if p >= 0.65 else "human",
                         "fusion_source": "cnn", "provenance_confidence": None, "watermark_confidence": None,
                         "cnn_confidence": max(p, 1 - p), "fusion_confidence": max(p, 1 - p)})
    out = report.by_generator(rows, 15)
    cnn = {r["generator"]: r for r in out if r["system"] == "cnn_only"}
    assert cnn["flux"]["recall"] == pytest.approx(0.8) and cnn["f_lite"]["recall"] == pytest.approx(0.2)
    assert cnn["all"]["recall"] == pytest.approx(0.5) and cnn["flux"]["n_human"] == 10
