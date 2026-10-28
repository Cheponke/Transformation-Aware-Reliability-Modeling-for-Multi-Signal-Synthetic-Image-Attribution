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
### Implemented in the current version as of October 28, 2026

- **src/attribute/evaluation/statistics.py** is subject to be changed. 

- **Dataset Preparation:** Dataset inspection, label handling, exploratory data analysis, train/validation/test split, and PyTorch dataset loading.
- **Image Transformation:**  Utilities for simulating image transformations such as social-media processing, JPEG compression, WebP conversion, resizing, and recompression.
- **Provenance Signal:** C2PA credential creation, signing, and verification utilities.
- **Watermark Signal:** Keyed additive watermark embedding and correlation-based detection.
- **Forensic CNN:** CNN model architectures, training utilities, inference, and temperature-based probability calibration.
- **Multi-Signal Fusion:** Hierarchical decision logic that prioritizes valid provenance, then watermark detection, and finally the calibrated CNN, with an uncertainty option.
- **Evaluation Metrics:** Accuracy, precision, recall, f1-score, False-Positive rate(FPR), Accuracy Degradation, Expected Calibration Error (ECE), and decision coverage.
- **Evaluation pipeline:** `scripts/evaluation.py` compares provenance_only, watermark_only, CNN_only, and multi-signal systems across configured transformation conditions. It generates per-image records and summary reports in CSV and JSON format.
- **Testing infrastructure:** Automated tests are provided for data handling, evaluation metrics, fusion logic, provenance utilities, and other implemented components.

### Current limitations and verification status
- Model trained checkpoints and calibration files must be generated before the full evaluation.
- C2PA dependencies and end-to-end signing and verification must be tested.
- The Experimental evaluation script must be eexecuted to generate actual results for each system and transformation condition.
- Statistical analysis tests specified in the manuscript must be implemented and connected to the evaluation outputs before statistical significance can be reported.
- **Results and reproducibility** Final metrics, comparisons, plots, and thesis tables must be generated from recorded experimental runs. No performance claims should be made without supporting results.
- Not implemented yet: the full evaluation run (`scripts/evaluate.py`). The provenance test needs `pip install -e ".[provenance]"` (c2pa-python).
  
- Datasets: Kannan AI vs Human Art, Superpotato9 AI recognition, AI-ArtBench.
