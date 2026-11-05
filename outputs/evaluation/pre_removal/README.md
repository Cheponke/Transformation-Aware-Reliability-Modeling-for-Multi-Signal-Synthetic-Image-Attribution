# Near-duplicate sensitivity before the removal (notebook 05b)

These files come from notebook 05b, run on 2026-10-29 with the models trained **before** the near-duplicate
removal. They are **not** results of the final models.

They are kept because they are the evidence that led to the removal. The near-duplicate check
(`outputs/data_quality/near_duplicates_summary.json`) found 157 pairs of near-duplicate images (dHash
Hamming distance ≤ 5) shared across the train, validation and test splits. 51 of the 2,992 test images had
a near-duplicate in another split. These files show each system's results on the full test split and with
those 51 images excluded (`*_all`, `*_without`, `*_change`).

After this run, notebook 01c removed the leaking copy of each pair from train and validation (113 train
and 12 validation images, `outputs/data_quality/removed_near_duplicates_summary.json`). A re-check found
no near-duplicates left (`outputs/data_quality/after_removal/`). All models were then retrained, and every
result outside this folder comes from the retrained models.
