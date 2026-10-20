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
  signals/                  provenance (placeholder), watermark, forensic CNN inference
  models/                   backbones, training loop, temperature scaling
  fusion/                   sequential decision logic with abstention
  evaluation/               accuracy/precision/recall/F1, FPR, ΔA, ECE, coverage
scripts/                    prepare_data.py, train.py, evaluate.py
tests/                      smoke tests
```

## Setup

```bash
python -m pip install -e ".[train,dev]"
pytest
```

Place each dataset under `data/raw/<name>/` with one folder per class (see `CLASS_DIRS` in
`src/attribution/data/datasets.py`), then:

```bash
python scripts/prepare_data.py
python scripts/train.py --arch resnet50
```

## Status

- Not implemented yet: C2PA signing and verification (`signals/provenance.py`) and the full evaluation run (`scripts/evaluate.py`).
- Datasets: Kannan AI vs Human Art, Superpotato9 AI recognition, AI-ArtBench.
