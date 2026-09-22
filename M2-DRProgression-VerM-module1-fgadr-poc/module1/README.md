# Module 1 — Severity Staging & Lesion Segmentation (DRG-Net)

Proof-of-concept implementation of the manuscript's Module 1 (Sec 3.2): joint DR severity
grading + pixel-level lesion segmentation, feeding a grade + lesion mask + Lesion Burden
Score (LBS) forward into Module 2's GAN conditioning.

**This does not reimplement DRG-Net.** It clones and runs the official reference
implementation directly:
https://github.com/DFKI-Interactive-Machine-Learning/dr-joint-learning
(Tusfiqur et al., arXiv:2212.14615 — CC BY-NC-SA 4.0, non-commercial academic use, cite the
paper). Pinned commit: `0da1bbe885f438390a0e94ec486283d9b21d5547`.

This folder only holds the thin adapter layer needed to point that code at *our* local
copies of FGADR (`../FGADR-Seg-set_Release/`) and IDRiD (`../IDRiD_dataset/`) instead of the
original authors' filesystem paths, plus the pieces DRG-Net doesn't provide at all (Lesion
Burden Score, and the bridge into Module 2's `fire_dataset.py`/`longdr_dataset.py`).

## Layout

- `dataprep/make_fgadr_classification_pkl.py`, `dataprep/make_idrid_classification_pkl.py`
  — build the `{'train','val','test': [(image_path, grade)]}` pickle DRG-Net's
  `dr_classification/main.py` expects. Adapted from DRG-Net's own
  `external/{fgadr,idrid}_generate_pkl.py` (same train/val/test split logic — a literal
  index slice for FGADR, train==val==test-source for IDRiD, exactly as published — only the
  paths and a pandas-version compatibility fix (`.iloc` instead of positional `row[0]`) changed.
- `configs/fgadr_poc.yaml`, `configs/idrid_poc.yaml` — copies of DRG-Net's own
  `dr_classification/configs/{fgadr,idrid}.yaml`, only `base.data_index`/`save_path`/
  `log_path`/`device` changed. **Epoch counts (20 / 60) are untouched from the paper** —
  small enough to run as-published on a Colab GPU.
- `data/DR_Seg_Grading_Label_Filtered.csv` — vendored as-is from DRG-Net's own
  `data/DR_Seg_Grading_Label_Filtered.csv` (pinned commit). This is metadata only (filename +
  grade + which images have all 4 lesion masks present), not FGADR's images — 401 of FGADR's
  1,842 images have complete MA/HE/EX/SE mask coverage; DRG-Net's segmentation pipeline
  (`get_images_fgadr_from_pd` in `dr_segmentation/utils.py`) reads this file directly from
  inside the FGADR directory it's pointed at, so the data-prep notebook copies it into the
  local FGADR folder. Verified all 401 rows resolve to real local files (image + all 4 masks).
- `configs/config_fgadr_seg_poc.py` — copy of DRG-Net's `dr_segmentation/config_fgadr.py`,
  `IMAGE_DIR` pointed locally, **`EPOCHES` reduced from 1500 → 25** (explicit, documented
  time-boxing — this is the honest answer to "why not 1500 epochs" for the presentation).
- `compute_lbs.py` — **ours, not part of DRG-Net**: `LBS = lesion_pixels / retinal_area_pixels`
  per the manuscript's formula, using a green-channel-threshold retinal FOV mask.
- `apply_to_progression_data.py` — loads a trained classification checkpoint + trained (or,
  for lesion classes not trained this round, ground-truth) segmentation masks, runs them over
  FIRE/LongDR baseline+follow-up images, and caches `{image_id: {mask, grade, lbs}}` so Module
  2 can consume real conditioning instead of the current placeholder.

## Known, deliberate scope limits (say these plainly if asked)

- Segmentation is trained for **2 of 4 lesion classes tonight (EX, MA)** — HE/SE use FGADR's
  own ground-truth masks as a stand-in in the bridge step, clearly labeled as such. Same
  script/config trains HE/SE next — just rerun with `--lesion HE` / `--lesion SE`.
- Segmentation epoch count is a **POC-scale reduction** from the paper's 1500/2500, not a
  claim of matching published performance.
- DRG-Net's own segmentation eval metric is **AP + ROC-AUC** (not Dice/IoU); we report both —
  AP/AUC to match the paper, Dice/IoU because the manuscript separately commits to that pair.
  `evaluate_segmentation_dice_iou.py` computes AP/ROC-AUC on the model's raw soft prediction
  (before thresholding), matching `train_fgadr.py::eval_model`'s own computation, so these are
  directly comparable to DRG-Net's Table III (FGADR-source, `S-Net+PD-Net+TATL`, the closest
  published setting to ours):

  | Lesion | Published AUC-ROC | Published AUC-PR |
  |---|---|---|
  | MA | 0.937 | 0.417 |
  | HE | 0.963 | 0.652 |
  | EX | 0.970 | 0.714 |
  | SE | 0.954 | 0.553 |

  `LESION_TO_FOLDER` (EX→HardExudate, HE→Hemohedge, MA→Microaneurysms, SE→SoftExudate) matches
  FGADR's own mask folder names and DRG-Net's own `LESION_IDS`, so there is no label-swap risk
  in this comparison.
- Classification eval reports **only Accuracy and Quadratic Weighted Kappa**, matching what
  DRG-Net's paper reports (Section VI-B) — sensitivity/specificity/F1 are deliberately not
  computed, since there is no published number to compare them against.
