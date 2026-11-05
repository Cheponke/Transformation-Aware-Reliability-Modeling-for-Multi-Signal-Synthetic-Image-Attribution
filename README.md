# Transformation-Aware Reliability Modeling for Multi-Signal Synthetic Image Attribution

Thesis project (Mapua Institute of Technology, BS Computer Science) by Jan Christian G. Diesta,
Michael Edward L. Gallor II and Guillian H. Hojilla.

The study measures how reliable three AI-art attribution signals stay under simulated
social-media processing, and whether a calibrated multi-signal framework beats each one alone:

1. **Provenance**: C2PA content credentials (hard binding to the file, plus a perceptual-hash soft binding)
2. **Watermark**: additive keyed pattern, `I_w = I + αW`, detected by correlation
3. **Forensic CNN**: ResNet-50 and EfficientNet-B0 (ResNet-18 as a compression baseline), temperature-calibrated

Decisions fall back in that order, and the system abstains ("uncertain") when no signal is present and the
CNN's probability is inside the abstention band (0.35–0.65 by default).

## Layout

```
configs/default.yaml        all experiment settings from the manuscript
src/attribution/
  data/                     dataset indexing, class-balanced sampling, 70/15/15 split, grouped k-fold,
                            EDA and near-duplicate (dHash) detection, torch Dataset
  transforms/               harmonization; JPEG / WebP / downscale / recompression conditions
  signals/                  provenance (C2PA signing/verification, perceptual hash), watermark, forensic CNN inference
  models/                   backbones, training loop, temperature scaling
  fusion/                   hierarchical decision logic with abstention
  evaluation/               metrics (accuracy, precision, recall, F1, FPR, ΔA, ECE, coverage) and statistics
                            (paired t-test, Nadeau–Bengio corrected t-test, Holm–Bonferroni)
scripts/                    command-line entry points (see the table below)
notebooks/                  Colab notebooks, run in the order below
tests/                      unit and smoke tests (pytest)
outputs/                    summary results committed from the experiment runs (see "Results")
```

## Setup

```bash
python -m pip install -e ".[train,dev]"        # add ,provenance for the C2PA experiments (c2pa-python)
pytest
```

## Running the experiments

The experiments were run on Google Colab with the notebooks below, in order. Each notebook saves its
results to `MyDrive/thesis-data/` so a later session can continue from them.

| Notebook | What it does | Main script(s) | Manuscript |
|---|---|---|---|
| `01_data_setup_and_eda` | Download the three Kaggle datasets, label, sample, split 70/15/15, harmonize to 256 px JPEG, EDA | `download_datasets.py`, `inspect_datasets.py`, `prepare_data.py`, `harmonize.py`, `eda.py` | 3.3–3.5 |
| `01b_near_duplicate_check` | Find near-duplicate images across train/val/test (dHash, Hamming ≤ 5) | `check_near_duplicates.py` | 3.5 |
| `01c_remove_near_duplicates` | Remove the leaking copy of each cross-split pair, then re-check | `remove_near_duplicates.py` | 3.5 |
| `02_resnet18_baseline` | Train and temperature-calibrate the CNNs; CNN compression stress test | `train.py`, `stress_test.py` | 3.6.3, 3.7.2 |
| `03_watermark` | Watermark detection threshold and stress test | `watermark_test.py` | 3.6.1 |
| `04_provenance` | C2PA signing, verification and soft-binding recovery under each transform | `provenance_test.py` | 3.6.2 |
| `05_fusion` | Evaluate provenance-only, watermark-only, CNN-only and multi-signal, with and without the watermark | `evaluate.py` | 3.7 |
| `05b_near_duplicate_sensitivity` | Same evaluation with near-duplicate test images excluded (run before the removal in 01c) | `evaluate.py --exclude-near-duplicates` | 3.5 |
| `05c_adoption_sweep` | Results when only part of the AI images carry a watermark / C2PA credential | `adoption_sweep.py` | 3.7 |
| `07_cross_validation` | Grouped 5-fold cross-validation and significance tests | `cross_validate.py` | 3.8.1 |
| `08_external_and_gradcam` | Unseen-generator test on WikiArt_VLM; Grad-CAM attention stability | `prepare_external.py`, `evaluate.py --split-csv`, `external_report.py`, `gradcam.py` | 3.7.3, 3.8.2, 3.8.3 |
| `06_collect_results` | Collect the summary files from Drive into `outputs/` for committing | – | – |

From a terminal, the core pipeline can also be run in one go (finished steps are skipped, so it is safe
to re-run after a crash):

```bash
python scripts/run_all.py                  # ResNet-18 baseline: data, EDA, training, stress test
python scripts/run_all.py --arch resnet50  # another backbone, reusing the prepared data
python scripts/run_all.py --from train     # force training and later steps to run again
```

Every script documents its options with `--help`.

## Data

- **Training/test pool:** Kannan *AI vs Human Art*, Superpotato9 *AI recognition*, and *AI-ArtBench* (Kaggle).
  Labels come from the `classes` glob rules per dataset in `configs/default.yaml`; check them with
  `inspect_datasets.py` after downloading. Near-duplicates across splits were removed before the final training.
- **External test:** *WikiArt_VLM* (Fu et al., 2026; Zenodo [10.5281/zenodo.17880956](https://doi.org/10.5281/zenodo.17880956), CC BY 4.0).
  WikiArt paintings paired with images generated from a caption of each painting by FLUX.1-dev and F-Lite
  (not in the training data) and Stable Diffusion 3.5 (a newer version of a family that is in the training data).
  `prepare_external.py` samples paintings so each appears once as a generated image and once as the original,
  harmonizes them like the training data, and removes any image that is a near-duplicate of a training-pool image.

## Results

The summary files from every notebook are in `outputs/` (model checkpoints and per-image CSVs are kept on
Drive only; `.gitignore` lets just the summaries in). `outputs/MANIFEST.json` lists each file and when it was produced.

| Folder | Contents |
|---|---|
| `outputs/data_quality/` | Near-duplicate pairs found and removed |
| `outputs/evaluation/pre_removal/` | Notebook 05b run before the near-duplicate removal (a record of why it was done, not final results) |
| `outputs/models/`, `outputs/stress_test/` | Calibration temperatures, training curves, CNN accuracy under each transform |
| `outputs/watermark/`, `outputs/provenance/` | Watermark threshold and survival; C2PA verification and soft-binding results |
| `outputs/evaluation/<arch>[_no_watermark]/` | The four systems on the test split, per transform |
| `outputs/evaluation/adoption/` | Watermark and C2PA adoption sweep, including where fusion overtakes watermark-only |
| `outputs/cv/` | Per-fold and mean ± SD results, paired and corrected t-tests with Holm adjustment |
| `outputs/evaluation/external/` | Unseen-generator results, per generator and against the in-distribution results |
| `outputs/gradcam/` | Grad-CAM example figures and attention-stability tables |

Metric conventions: recall and FPR are computed over all images (an abstention counts as not flagged);
accuracy, precision and ECE are computed on the images a system decides; coverage is the share decided.

## Status (as of November 4, 2026)

All experiments in the manuscript's methodology have been implemented and run:

- Data preparation, EDA and near-duplicate removal
- CNN training with temperature calibration (ResNet-18, ResNet-50, EfficientNet-B0) and compression stress test
- Watermark and C2PA provenance stress tests
- Multi-signal fusion evaluation, with and without the watermark, and with near-duplicates excluded
- Watermark / C2PA adoption sweep
- Grouped 5-fold cross-validation with significance tests
- External unseen-generator test and Grad-CAM attention stability

## Limitations

- **Provenance and watermark are applied by the study itself.** Every test image is signed with a test
  certificate and every AI image is watermarked with the study's key, so on unaltered images these signals are
  correct by construction. What the experiments measure is how well they survive processing and what happens when
  they are missing; the adoption sweep covers partial adoption. This also means the external test mainly tests
  the CNN, since the other two signals don't depend on which generator made the image.
- **The watermark is a simple additive scheme,** not a production watermark (for example a learned one embedded
  during generation). Results for it should not be read as results for deployed watermarking systems.
- **Embedding the watermark lowers the CNN's AI probability** on AI images (the CNNs were trained on unmarked
  images), so the CNN-only numbers in the watermarked runs are lower than in the no-watermark runs.
- **The external test covers one domain** (WikiArt-style paintings) and three generators.
- **The external test and Grad-CAM use the single-split models,** not the cross-validation fold models.
- **Transforms are simulated** (JPEG, WebP, resizing, repeated recompression), not uploads to real platforms.
