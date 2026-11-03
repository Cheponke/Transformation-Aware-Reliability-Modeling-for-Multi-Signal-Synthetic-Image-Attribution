"""Grad-CAM explanations for the CNN channel (manuscript sections 3.7.3 and 3.8.3).

L_c = ReLU(sum_k alpha_k^c A^k), with alpha_k^c the spatial mean of d y_c / d A^k
(Selvaraju et al., 2017), on the last convolutional block (ResNet-50 layer4,
EfficientNet-B0 features). The single logit is explained toward the predicted
class: +logit when the model says AI, -logit when it says human.

Produces, per model:
- examples_<set>.png: test images (correct and wrong, each class; per generator for
  the external set) with their heatmap on the clean image and after JPEG q30 and
  0.5x downscaling;
- watermark.png: AI images with and without the generation-time watermark, with
  both heatmaps and P(AI), to show how the watermark changes what the CNN uses;
- stability.csv / stability_summary.json: on a sample of images, how much of the
  top-20% attention region survives each transformation (IoU with the clean
  heatmap), how often the prediction flips, and the same for watermarked vs
  unmarked AI images.

Writes outputs/gradcam/<arch>/. Usage: python scripts/gradcam.py --arch efficientnet_b0
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from attribution.config import load_config, splits_dir
from attribution.data.splits import read_split
from attribution.data.torch_dataset import eval_transform
from attribution.models.cnn import build_model
from attribution.signals import watermark
from attribution.transforms.social_media import build_conditions

TARGET_LAYERS = {"resnet18": "layer4", "resnet50": "layer4", "efficientnet_b0": "features"}
TOP_FRACTION = 0.2


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer_name: str):
        self.model = model
        layer = model
        for part in layer_name.split("."):
            layer = getattr(layer, part)
        self._acts = self._grads = None

        def keep(module, inputs, output):
            # a hook on the tensor itself is safe with the in-place activations inside the blocks
            self._acts = output
            output.register_hook(lambda grad: setattr(self, "_grads", grad))

        layer.register_forward_hook(keep)

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Heatmaps in [0, 1] at the input size, and logits, for a batch."""
        self.model.zero_grad(set_to_none=True)
        x = x.requires_grad_(True)
        logits = self.model(x).squeeze(1)
        # explain each image toward its own predicted class; images don't interact in eval mode
        (torch.where(logits >= 0, logits, -logits)).sum().backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        flat = cam.flatten(1)
        lo, hi = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
        cam = ((flat - lo) / (hi - lo).clamp_min(1e-8)).view_as(cam)
        return cam.detach().cpu().numpy(), logits.detach().cpu().numpy()


def top_mask(cam: np.ndarray, fraction: float = TOP_FRACTION) -> np.ndarray:
    return cam >= np.quantile(cam, 1 - fraction)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def _load(path: str) -> Image.Image:
    with Image.open(path) as f:
        return f.convert("RGB")


def explain(cam_fn, images: list[Image.Image], cfg: dict, device: str, temperature: float, batch: int = 32):
    tf = eval_transform(cfg)
    cams, probs = [], []
    for start in range(0, len(images), batch):
        x = torch.stack([tf(img) for img in images[start:start + batch]]).to(device)
        c, logits = cam_fn(x)
        cams.append(c)
        probs.append(1 / (1 + np.exp(-logits / temperature)))
    return np.concatenate(cams), np.concatenate(probs)


def _display(img: Image.Image, cfg: dict) -> np.ndarray:
    size = int(cfg["preprocess"]["image_size"])
    return np.asarray(img.resize((size, size), Image.BILINEAR))


def example_figure(path: Path, title: str, rows: list[tuple[str, Image.Image]], cfg, cam_fn, device, temperature):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = build_conditions(cfg)
    columns = [("clean", None), ("JPEG q30", conditions["jpeg_q30"]), ("resize 0.5", conditions["resize_0.5"])]
    fig, axes = plt.subplots(len(rows), 1 + len(columns), figsize=(3 * (1 + len(columns)), 3 * len(rows)), squeeze=False)
    for ax_row, (label, img) in zip(axes, rows):
        variants = [img if t is None else t(img) for _, t in columns]
        cams, probs = explain(cam_fn, variants, cfg, device, temperature)
        ax_row[0].imshow(_display(img, cfg))
        ax_row[0].set_title(label, fontsize=8)
        for ax, (name, _), variant, cam, p in zip(ax_row[1:], columns, variants, cams, probs):
            ax.imshow(_display(variant, cfg))
            ax.imshow(cam, cmap="jet", alpha=0.45)
            ax.set_title(f"{name}: P(AI)={p:.2f}", fontsize=8)
        for ax in ax_row:
            ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=80)
    plt.close(fig)


def watermark_figure(path: Path, title: str, images: list[Image.Image], cfg, cam_fn, device, temperature):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alpha, key = float(cfg["watermark"]["alpha"]), int(cfg["watermark"]["key"])
    fig, axes = plt.subplots(len(images), 4, figsize=(12, 3 * len(images)), squeeze=False)
    for ax_row, img in zip(axes, images):
        marked = watermark.embed(img, alpha, key)
        cams, probs = explain(cam_fn, [img, marked], cfg, device, temperature)
        ax_row[0].imshow(_display(img, cfg))
        ax_row[0].set_title("AI image, unmarked", fontsize=8)
        for ax, variant, cam, p, name in zip(ax_row[1:3], [img, marked], cams, probs, ["unmarked", "watermarked"]):
            ax.imshow(_display(variant, cfg))
            ax.imshow(cam, cmap="jet", alpha=0.45)
            ax.set_title(f"{name}: P(AI)={p:.2f}", fontsize=8)
        ax_row[3].imshow(cams[1] - cams[0], cmap="coolwarm", vmin=-1, vmax=1)
        ax_row[3].set_title(f"attention change (IoU top-20%: {iou(top_mask(cams[0]), top_mask(cams[1])):.2f})", fontsize=8)
        for ax in ax_row:
            ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=80)
    plt.close(fig)


def stability(records, cfg, cam_fn, device, temperature, set_name: str) -> list[dict]:
    """IoU of the top-20% attention region between clean and transformed images, plus prediction flips."""
    conditions = {k: v for k, v in build_conditions(cfg).items() if v is not None}
    alpha, key = float(cfg["watermark"]["alpha"]), int(cfg["watermark"]["key"])
    images = [_load(r.path) for r in records]
    labels = np.array([r.label for r in records])
    base_cams, base_p = explain(cam_fn, images, cfg, device, temperature)
    base_masks = [top_mask(c) for c in base_cams]
    rows = []
    for name, t in conditions.items():
        cams, p = explain(cam_fn, [t(img) for img in images], cfg, device, temperature)
        ious = np.array([iou(m, top_mask(c)) for m, c in zip(base_masks, cams)])
        flips = (base_p >= 0.5) != (p >= 0.5)
        for lab, cls in [(0, "human"), (1, "ai")]:
            sel = labels == lab
            rows.append({"set": set_name, "comparison": f"clean vs {name}", "class": cls, "n": int(sel.sum()),
                         "iou_top20_mean": float(ious[sel].mean()), "iou_top20_sd": float(ious[sel].std()),
                         "prediction_flip_rate": float(flips[sel].mean()),
                         "mean_p_ai_before": float(base_p[sel].mean()), "mean_p_ai_after": float(p[sel].mean())})
    ai_idx = np.flatnonzero(labels == 1)
    if len(ai_idx):
        marked = [watermark.embed(images[i], alpha, key) for i in ai_idx]
        cams, p = explain(cam_fn, marked, cfg, device, temperature)
        ious = np.array([iou(base_masks[i], top_mask(c)) for i, c in zip(ai_idx, cams)])
        rows.append({"set": set_name, "comparison": "unmarked vs watermarked", "class": "ai", "n": len(ai_idx),
                     "iou_top20_mean": float(ious.mean()), "iou_top20_sd": float(ious.std()),
                     "prediction_flip_rate": float(((base_p[ai_idx] >= 0.5) != (p >= 0.5)).mean()),
                     "mean_p_ai_before": float(base_p[ai_idx].mean()), "mean_p_ai_after": float(p.mean())})
    return rows


def pick_examples(records, probs, rng, per_cell: int, by_source: bool):
    """Correct and wrong examples of each class (and of each generator when by_source)."""
    out = []
    groups = sorted({r.source for r in records if r.label == 1}) if by_source else [None]
    for label, cls in [(0, "human"), (1, "AI")]:
        for g in (groups if label == 1 else [None]):
            for correct in (True, False):
                pool = [(r, p) for r, p in zip(records, probs) if r.label == label and (g is None or r.source == g)
                        and ((p >= 0.5) == bool(label)) == correct]
                for r, p in rng.sample(pool, min(per_cell, len(pool))):
                    tag = f"{cls}{'/' + g if g else ''} ({'correct' if correct else 'wrong'})"
                    out.append((tag, _load(r.path)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--models", default="outputs/models")
    ap.add_argument("--external-csv", default="data/external/external.csv")
    ap.add_argument("--out", default="outputs/gradcam")
    ap.add_argument("--n-stability", type=int, default=300, help="images per class for the stability numbers")
    ap.add_argument("--per-cell", type=int, default=2, help="examples per (class, correct/wrong) in the figures")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    rng = random.Random(int(cfg["seed"]))

    model = build_model(args.arch, pretrained=False, dropout=cfg["model"]["dropout"])
    model.load_state_dict(torch.load(Path(args.models) / args.arch / "best.pt", map_location=device))
    model.to(device).eval()
    temperature = float(json.loads((Path(args.models) / args.arch / "calibration.json").read_text())["temperature"])
    cam_fn = GradCAM(model, TARGET_LAYERS[args.arch])
    out = Path(args.out) / args.arch
    out.mkdir(parents=True, exist_ok=True)

    sets = {"in_distribution": read_split(splits_dir(cfg) / "test.csv")}
    if Path(args.external_csv).is_file():
        sets["external"] = read_split(Path(args.external_csv))
    else:
        print(f"no external set at {args.external_csv}: in-distribution only")

    stability_rows = []
    for name, records in sets.items():
        sample = []
        for label in (0, 1):
            group = [r for r in records if r.label == label]
            sample += rng.sample(group, min(args.n_stability, len(group)))
        _, probs = explain(cam_fn, [_load(r.path) for r in sample], cfg, device, temperature)
        examples = pick_examples(sample, probs, rng, args.per_cell, by_source=name == "external")
        example_figure(out / f"examples_{name}.png", f"{args.arch}: Grad-CAM, {name.replace('_', ' ')} test images",
                       examples, cfg, cam_fn, device, temperature)
        print(f"{name}: wrote {out / f'examples_{name}.png'} ({len(examples)} images)", flush=True)
        stability_rows += stability(sample, cfg, cam_fn, device, temperature, name)
        if name == "in_distribution":
            ai = [r for r, p in zip(sample, probs) if r.label == 1][:6]
            watermark_figure(out / "watermark.png", f"{args.arch}: what the watermark changes in the CNN's attention",
                             [_load(r.path) for r in ai], cfg, cam_fn, device, temperature)
            print(f"wrote {out / 'watermark.png'}", flush=True)

    with (out / "stability.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stability_rows[0]))
        w.writeheader()
        w.writerows(stability_rows)
    (out / "stability_summary.json").write_text(json.dumps({
        "arch": args.arch, "target_layer": TARGET_LAYERS[args.arch], "top_fraction": TOP_FRACTION,
        "n_per_class": args.n_stability, "sets": list(sets), "rows": stability_rows}, indent=1))
    for r in stability_rows:
        if r["comparison"] in ("clean vs jpeg_q30", "clean vs resize_0.5", "unmarked vs watermarked"):
            print(f"  {r['set']:>15} {r['comparison']:>24} {r['class']:>5}: IoU {r['iou_top20_mean']:.2f}, "
                  f"flips {r['prediction_flip_rate']:.1%}, P(AI) {r['mean_p_ai_before']:.2f} -> {r['mean_p_ai_after']:.2f}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
