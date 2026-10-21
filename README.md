# Transformation-Aware Reliability Modeling for Multi-Signal Synthetic Image Attribution

Thesis project (Mapua Institute of Technology, BS Computer Science) by Jan Christian G. Diesta,
Michael Edward L. Gallor II and Guillian H. Hojilla.

The study measures how reliable three AI-art attribution signals stay under simulated
social-media processing, and whether a calibrated multi-signal framework beats each one alone:

1. **Provenance**: C2PA content credentials
2. **Watermark**: additive keyed pattern, `I_w = I + αW`, detected by correlation
3. **Forensic CNN**: ResNet-50 and EfficientNet-B0 (ResNet-18 as a compression baseline), temperature-calibrated

## Run

Everything is in one file, `pipeline.py`: download the three Kaggle datasets, label and sample
10k images per class, split 70/15/15, harmonize to 256 px JPEG, EDA, train, then the compression
stress test (clean vs JPEG 90/70/50/30, WebP, downscaling, recompression).

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell; on Linux/Mac: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU
pip install pillow numpy kaggle
python pipeline.py
```

Put your Kaggle token at `~/.kaggle/kaggle.json` first. Finished steps are skipped, so after a
crash just run `python pipeline.py` again. Other options:

```bash
python pipeline.py --arch resnet50     # another backbone on the same data
python pipeline.py --from train        # re-run training and the stress test
```

Results: `outputs/stress_test/<arch>/results.csv` (accuracy, precision, recall, F1, FPR, ECE, ΔA per
condition), training curves in `outputs/models/<arch>/history.json`, EDA in `outputs/eda/report.json`.
All settings are constants at the top of `pipeline.py`.

On Colab: `!git clone <this repo> && cd` into it, upload `kaggle.json` to `~/.kaggle/`, then `!python pipeline.py`.

## Not implemented yet

C2PA provenance, the watermark channel and the multi-signal fusion. The earlier modules for the
watermark and fusion are in git history (before the single-file rewrite) if needed.
