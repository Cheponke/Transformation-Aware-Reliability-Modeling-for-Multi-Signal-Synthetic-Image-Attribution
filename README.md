# Transformation-Aware Reliability Modeling for Multi-Signal Synthetic Image Attribution

Thesis project (Mapua Institute of Technology, BS Computer Science) by Jan Christian G. Diesta,
Michael Edward L. Gallor II and Guillian H. Hojilla.

The study measures how reliable three AI-art attribution signals stay under simulated
social-media processing, and whether a calibrated multi-signal framework beats each one alone:

1. **Provenance**: C2PA content credentials
2. **Watermark**: additive keyed pattern, `I_w = I + αW`, detected by correlation
3. **Forensic CNN**: ResNet-50 and EfficientNet-B0 (ResNet-18 as a compression baseline), temperature-calibrated

Decisions fall back in that order, and the system abstains ("uncertain") when the CNN is not confident.

## Layout

```
configs/default.yaml        all experiment settings from the manuscript
src/attribution/
  data/                     dataset indexing, class-balanced sampling, 70/15/15 + k-fold splits, torch Dataset
  transforms/               JPEG / WebP / downscale / recompression conditions
  signals/                  provenance (C2PA signing/verification, perceptual hash), watermark, forensic CNN inference
  models/                   backbones, training loop, temperature scaling
  fusion/                   sequential decision logic with abstention
  evaluation/               accuracy/precision/recall/F1, FPR, ΔA, ECE, coverage
scripts/                    download_datasets.py, inspect_datasets.py, prepare_data.py, harmonize.py, eda.py, train.py, stress_test.py, watermark_test.py, provenance_test.py, run_all.py, evaluate.py
notebooks/                  Colab notebooks (01: data setup and EDA, 02: CNN training + compression stress test, 03: watermark stress test, 04: provenance stress test)
tests/                      smoke tests
```

## Setup

```bash
python -m pip install -e ".[train,dev]"
pytest
```

The easiest route is `notebooks/01_data_setup_and_eda.ipynb` on Colab. From a terminal:

```bash
python scripts/download_datasets.py   # needs ~/.kaggle/kaggle.json
python scripts/inspect_datasets.py    # every folder should be labeled human or ai
python scripts/prepare_data.py        # writes data/processed/{train,val,test}.csv
python scripts/harmonize.py           # 256 px JPEG copies in data/harmonized/ (used for training)
python scripts/eda.py                 # writes outputs/eda/report.json
python scripts/train.py --arch resnet50
```

Or run everything in one go (finished steps are skipped, so it is safe to re-run after a crash):

```bash
python scripts/run_all.py                  # ResNet-18 baseline: data, EDA, training, stress test
python scripts/run_all.py --arch resnet50  # another backbone, reusing the prepared data
python scripts/run_all.py --from train     # force training and later steps to run again
```

Labels come from the `classes` glob rules per dataset in `configs/default.yaml`. The Kaggle slugs and
folder patterns there are unverified until the first download, so check them with `inspect_datasets.py`.

## Status

- Not implemented yet: the full evaluation run (`scripts/evaluate.py`). The provenance test needs `pip install -e ".[provenance]"` (c2pa-python).
- Datasets: Kannan AI vs Human Art, Superpotato9 AI recognition, AI-ArtBench.
