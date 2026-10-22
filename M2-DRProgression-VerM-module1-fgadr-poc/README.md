# TO-DO LIST
[] Module 1
[] Module 2
[] Module 3

# Branches
- *Feilong-DRForestGAN* is the original code found via https://zenodo.org/records/15747872.
- *module-2-inference*
- *module-2-batch-inference*

# Dataset
As of May 3, 2026, I used the following dataset to try to apply to *module-2-inference* branch.
You can download the archive.zip through this link: https://www.kaggle.com/datasets/lawateaditya/idrid-dataset?resource=download
It is too big for Github.

As of May 5, I used the ff link for the FIRE dataset: https://projects.ics.forth.gr/cvrl/fire/
and Longitudinal DR dataset: https://academictorrents.com/details/744717095e59373186abec814c86de4831d889e9

# Updates
May 3: inference preliminary testing is a success.
May 5: batch inference (with 5 images) preliminary testing is a success.

# Module 2 FID comparison -- sentence to use in the manuscript/defense
FID (feature=2048) is biased at small sample sizes, not just noisy (Binkowski et al. 2018),
so `train_module2_poc.py`/`evaluate_trajectory.py` also report a bootstrap 95% range and
Kernel Inception Distance (KID, unbiased at any N) alongside the point estimate -- see
`image_quality_metrics.py`. Frame the comparison directionally, not competitively:

> "Our FID is computed on a substantially smaller held-out set than DRForecastGAN's (n=X vs.
> their 2,734-8,523), so it indicates general range rather than a strict head-to-head result;
> KID is reported alongside as a sample-size-robust complement."

Say it once, plainly, rather than either overclaiming comparability or quietly reporting
`feature=64` and hoping it isn't questioned.
# Module 2 -- non-longitudinal (EyePACS) pretraining (branch `mod-2-NonLongIntegration`)
Our longitudinal pairs (FIRE + LongDR + Tianjin) are still small, so Module 2 is now trained in
two stages:

1. **Pretrain on a cross-sectional graded set** (EyePACS: ~35k images labelled ICDR 0-4; APTOS
   or any `image,grade` CSV also works). No pairs needed -- the Generator learns what each stage
   looks like (StarGAN-style stage classification + cycle/identity losses) and is pulled toward
   each stage's **biomarker profile** (per-lesion EX/HE/MA/SE area + count, and LBS, measured by
   Module 1 on every EyePACS image and grouped by the real grade).
2. **Fine-tune on the longitudinal pairs** with the existing `train_module2_poc.py`, starting from
   the pretrained weights (`--init-generator` / `--init-discriminator`).

Dataset: resized EyePACS, https://www.kaggle.com/datasets/mohlamin/resized-eyepacs-diabetic-retinopathy-dataset
(`all_labels.csv` = original EyePACS train + test labels merged, ICDR 0-4). Used for **both**
Module 1 (extra grading data) and Module 2 (cross-sectional pretraining). The loaders find
`all_labels.csv` anywhere under the extracted folder, auto-detect its image/grade columns, and
index images recursively by file name, so the zip's folder layout doesn't matter. Research use
only under the original EyePACS/Kaggle terms -- don't redistribute the images.

```bash
# Module 1: EyePACS grading pickle (patient-level split -- both eyes of a patient stay together)
python module1/dataprep/make_eyepacs_classification_pkl.py --eyepacs-root /content/data/eyepacs \
    --out /content/drive/MyDrive/Thesis_Datasets/module1_pkl/eyepacs_classification_pkl.pkl
#   (add --combine-with .../fgadr_classification_pkl.pkl to train on FGADR + EyePACS together)
# then, in DRG-Net's dr_classification/:  python main.py -c configs/eyepacs_poc.yaml
```

Module 2 pipeline:

```bash
# a) Module 1 over EyePACS -- now also stores per-lesion lesion_stats; --mask-size keeps the cache small
python module1/apply_to_progression_data.py --images-dir /content/data/eyepacs \
    --classifier-checkpoint .../final_weights.pt \
    --seg-checkpoint EX=... --seg-checkpoint HE=... --seg-checkpoint MA=... --seg-checkpoint SE=... \
    --mask-size 256 --out /content/drive/MyDrive/.../module1_outputs_eyepacs.pt

# b) Which biomarkers characterise each stage / shift between adjacent stages
python stage_biomarker_profiles.py --module1-cache .../module1_outputs_eyepacs.pt \
    --labels-csv /content/data/eyepacs/all_labels.csv \
    --out ./stage_biomarker_profiles.json --fig ./stage_biomarker_profiles.png

# c) Stage-1 pretraining on EyePACS
python train_module2_nonlong_pretrain.py --dataset-dir /content/data/eyepacs --preset eyepacs_resized \
    --module1-cache .../module1_outputs_eyepacs.pt --profiles ./stage_biomarker_profiles.json \
    --save-dir ./DRForestGAN-v2/stargan/models_nonlong/

# d) Stage-2 longitudinal fine-tuning (same flags as before + the init checkpoints)
python train_module2_poc.py --tianjin-dir ... --tianjin-module1-cache ... \
    --fire-module1-cache ... --longdr-module1-cache ... \
    --init-generator ./DRForestGAN-v2/stargan/models_nonlong/final-G.ckpt \
    --init-discriminator ./DRForestGAN-v2/stargan/models_nonlong/final-D.ckpt \
    --save-dir ./DRForestGAN-v2/stargan/models_poc_from_nonlong/
```

**Best-epoch saving (step d and Module 3).** `train_module2_poc.py` holds out 10% of the
eyes/patients (`--val-fraction`, split by eye/patient so no eye is on both sides), scores the
Generator on them after every epoch (L1 to the real follow-up) and writes the best epoch to
`best-G.ckpt`; `final-G.ckpt` is the last epoch. `module3/train_module3_poc.py` does the same on
its val patients (NLL) -> `best-model.ckpt` / `final-model.ckpt`. Notebooks 06-08 and 10 use the
`best-` checkpoints. The EyePACS pretraining (step c) keeps its last epoch: its only validation
signals come from the D/R networks trained alongside G, so they can't pick an epoch fairly, and
fine-tuning selects the epoch that is actually used.

`stage_biomarker_profiles.json` holds, per stage, each biomarker's distribution and the lesions
present in >=50% of eyes (`characteristic_lesions`); per transition `k->k+1`, the biomarkers ranked
by effect size (`ranked_shifts`, Cohen's d + Mann-Whitney p) and the ones that barely move
(`shared_biomarkers`, |d| < 0.2). Caveat for the write-up: these come from Module 1's *predicted*
masks, so they describe what Module 1 sees per stage, not manual annotations.

Without `--module1-cache/--profiles`, step (c) still runs as plain stage translation (biomarker
losses off). `python train_module2_nonlong_pretrain.py --dry-run --save-dir /tmp/nl` smoke-tests the
whole pipeline on synthetic data. Compare `models_poc_from_nonlong/` against a from-scratch
`train_module2_poc.py` run with `evaluate_trajectory.py` to show whether pretraining helps.

# Branch `AllModuleEdits-PreFinals` -- pre-finals fixes (`CODE_FIXES_AllModuleEdits-PreFinals.md`)
Branched from `mod-2-NonLongIntegration`. Every notebook clones this branch and writes to NEW Drive
folders (`RUN_SUFFIX = '_prefinals'` in the first code cell: `module1_cache_*_prefinals/`,
`module2_runs/*_prefinals`, `module3_runs/*_prefinals`, `module1_runs/drgnet_replication_prefinals/`),
so the earlier checkpoints, caches and results stay available for before/after comparison. Each
change that alters results keeps the earlier behaviour behind a flag:

| Item | Change (default now) | Earlier behaviour |
|---|---|---|
| 1.1 / 1.2 | Lesion segmentation on overlapping 512 px tiles of a ~1280 px retina crop (pipeline, `evaluate_segmentation_dice_iou.py`, seg-grid figure); grader also sees the retina crop; masks clipped to the field of view | `--seg-mode resize --no-retina-crop` |
| 1.3 / 1.4 | FGADR + DDR segmenters to 200 epochs (per-lesion `SEG_EPOCHS`, resumes from 100); soft Dice term for MA (`SEG_DICE_WEIGHT`, `DRG_SEG_DICE_WEIGHT`) | `SEG_EPOCHS` 100, `SEG_DICE_WEIGHT = {}` |
| 1.5 | Grader learning-curve cell fixed (bash syntax error) | -- |
| 1.7 | EyePACS pickle: blank `Usage` = training image (was dropped -> 0 train images); EyePACS grader trained (nb 11, progressively balanced) and used as `CLASSIFIER_RUN = 'eyepacs'` (falls back to FGADR's until it exists) | `CLASSIFIER_RUN = 'fgadr'` |
| 2.1 / 2.2 | Fine-tuning conditions on the FOLLOW-UP grade (Tianjin clinical; FIRE/LongDR Module 1's grade of the follow-up image) | `--condition-on baseline` |
| 2.3 | Stage classification loss in fine-tuning (D on clinically graded real follow-ups, G on its outputs) | `--lambda-cls 0` |
| 2.4 | Progressing pairs oversampled to 30 % of training samples | `--progress-fraction 0` |
| 2.5 | Pretraining evaluation was all grade-0 images (val ordered by grade + capped batches) -> 0 % stage-head accuracy; val shuffled, balanced accuracy + confusion reported | -- |
| 3.1-3.3 | `n_critic` 5, lr 1e-4 for G and D, fine-tuning batch 8 | `--n-critic 1 --g-lr 2e-4 --d-lr 2e-4 --batch-size 4` |
| 3.4 / 3.6 / 4.1 / 4.2 | Spectral-norm D; G InstanceNorm without running stats, Upsample + conv instead of ConvTranspose, U-Net skips | `--model-arch legacy` |
| 3.5 | Fine-tuning starts from a fresh D | `--init-discriminator-mode full` |
| 3.7 | `<save_dir>/batch_log.csv`: per-batch D loss, GP, gradient norms, G losses | -- |
| 4.3 | One `MODULE2_IMAGE_SIZE` (notebooks 06/08/09/10), checked against each checkpoint's run and the registration cache (still 128) | -- |
| 5.1 / 5.2 | Tianjin pair quality >= 0.5; registration validated on FIRE's ground-truth control points (identity / net / SIFT+RANSAC), `--method sift` option | `TIANJIN_MIN_PAIR_QUALITY = None` |
| 5.3 | Best epoch = lowest val L1 - 0.1 x Module 1 agreement with the target stage; early stopping (patience 25) | `--selection-alpha 0`, `--early-stop-patience 0` |
| 5.4-5.7 | Pretrained G evaluated as "epoch 0"; FID range by subsampling without replacement (+ no negative error bars); samples sorted numerically; ablation runs get their own folders | `range_method="bootstrap"` |
| 6.1-6.3 | Module 3: patient-level 70/15/15 split, metrics on TEST patients; first 6 EfficientNet-B4 blocks frozen; augmentation + AdamW weight decay 1e-2 | `--test-fraction 0 --val-fraction 0.2 --freeze-blocks 0 --no-augment --weight-decay 0` |
| 7.1 | Laterality: disc found inside the field of view, pair constraint, validated on EyePACS (nb 11) -- **check the OD/OS convention it reports** (`--disc-left-is`) | `--legacy-estimator --no-pair-constraint` |
| 7.2 / 7.3 | `MIN_BASELINE_GRADE` (0 = keep Stage 0, 1 = >= mild NPDR); `tools/dataset_image_sizes.py` | -- |

**Checkpoints:** items 3.4/3.6/4.1/4.2 change the Generator/Discriminator `state_dict`s, so
checkpoints from `mod-2-NonLongIntegration` (including the EyePACS pretraining) do NOT load into the
default models -- the scripts stop with an error naming `--model-arch legacy`, which loads them. The
whole Module 2 chain (09 pretraining -> 06 fine-tuning) must be retrained on this branch.

**Re-run order:** 11 (tiled re-evaluation, 200-epoch segmenters, EyePACS grader + laterality
validation) -> 04_05 (rebuild all Module 1 caches) -> 09 (EyePACS cache, profiles, pretraining) -> 06
(fine-tuning, trajectory evaluation, epoch-0 eval; then optionally the `USE_NONLONG_PRETRAIN = False`
ablation) -> 07 -> 08 -> 10.

# Notebooks -- what each runs, in order (Colab, `notebooks/`)
| # | Notebook | Runs | Needs |
|---|---|---|---|
| 01 | `01_module1_data_prep_colab` | FGADR + IDRiD grading pickles, sample LBS figure | dataset zips on Drive |
| 02 | `02_module1_classification_train_colab` | DRG-Net ResNet50 grading classifier (FGADR 20 ep, IDRiD 60 ep) | 01 |
| 03 | `03_module1_segmentation_train_colab` | DRG-Net U-Net lesion segmenters EX/MA/HE/SE (25 ep) + Dice/IoU/ROC figures | 01 |
| 04_05 | `04_05_combined_module1_caches_colab` | Module 1 over FIRE + LongDR + Tianjin -> `module1_outputs_*.pt` (replaces 04 + 05's caching) | 02, 03 |
| 04 / 05 | older split versions of 04_05 (04 also had a throwaway 5-epoch Module 2 run) | -- | -- |
| 06 | `06_module2_poc_with_tianjin_colab` | Module 2 training on FIRE + LongDR + Tianjin (registration, 50 ep) + `evaluate_trajectory.py` + figures; fine-tunes from 09's EyePACS pretraining (`USE_NONLONG_PRETRAIN = True`, default) | 04_05, 09 |
| 07 | `07_module3_poc_colab` | Module 3 (EfficientNet-B4 + Weibull) time-to-progression on Tianjin; run copied to Drive for 08 / 10 | 04_05 |
| 08 | `08_visualize_trajectory_colab` | Cascade figures for Tianjin patients (one per eye) + Module 3 captions; `END_STAGE = 4` goes to PDR; figures saved to Drive | 06, 07 |
| 09 | `09_eyepacs_module1_and_module2_pretrain_colab` | **new** -- EyePACS: Module 1 grading pickle (+ optional classifier training); Module 1 cache over an EyePACS subset; stage biomarker profiles; Module 2 non-longitudinal pretraining | 02, 03, EyePACS zip on Drive |
| 11 | `11_module1_drgnet_replication_colab` | **new** -- DRG-Net's own Module 1 runs, one model per dataset as in their repo: grading on FGADR (released split), IDRiD, EyePACS; lesion segmentation on FGADR and IDRiD (100 ep each by default; DRG-Net used 1500 / 2500), plus DDR (our addition, 100 ep, official train / val / test; `module1/dataprep/prepare_ddr_segmentation.py`) and FGADR + DDR combined (100 ep; `prepare_combined_segmentation.py`) -- the pipeline's lesion models; every model scored on the FGADR / DDR / IDRiD test sets; resumes from Drive checkpoints after a disconnect; test-set metrics summary | dataset zips on Drive (IDRiD "A. Segmentation" for IDRiD seg; a zip with DDR in its name for DDR seg) |
| 10 | `10_generate_progression_to_pdr_colab` | **new** -- any fundus image(s) -> Module 1 grade -> cascade to stage 4 (PDR), saves per-stage images/figure/JSON | 02, 03, 06 (07 optional) |

**`MODULE1_SOURCE`** (first code cell of 04_05 and 06-10) picks which Module 1 models the pipeline
uses. Use the same value in every notebook; each value writes to its own folders, so switching never
overwrites another setting's caches or models.

| `MODULE1_SOURCE` | Classifier | Lesion segmenters | Folder suffix |
|---|---|---|---|
| `'drgnet_combined'` (default) | FGADR (nb 11) | FGADR + DDR combined (nb 11, `comb_models_*`) | `_drgnet_comb` |
| `'drgnet'` | FGADR (nb 11) | FGADR only (nb 11, `models_FGADR_NO_TATL_*`) | `_drgnet` |
| `'poc'` | nb 02 | nb 03 | none |

Order with the DRG-Net Module 1 (default):
**11 -> 04_05 -> 09 -> 06 -> 07 -> 08 -> 10** (01-03 not needed; 11 builds its own data files).

Order with the POC Module 1 (`MODULE1_SOURCE = 'poc'`):
**01 -> 02 + 03 -> 04_05 -> 09 -> 06 -> 07 -> 08 -> 10**.

**Google Drive storage.** Every overwrite of a file on a Colab-mounted Drive leaves the old copy in
the Trash / version history, and those copies count against the Drive quota. Writing ~250 MB
checkpoints straight to Drive every epoch fills a 15 GB Drive within hours, after which model files
silently fail to upload (small logs still save). So all training writes checkpoints, resume states
and growing caches to the VM disk and copies them to Drive every N epochs and at the end
(`drive_io.py`, `resume_utils.py`, the mirror part of `module1/drgnet_resume_patch.py`); after each
run the notebooks check the model files actually reached Drive and stop with a message if not. Each
notebook prints the free Drive space after mounting, and its last cell (`drive.flush_and_unmount()`)
waits for uploads to finish -- run it before closing the runtime. Keep ~15 GB free and empty the
Drive Trash between long runs.

Notebook 06 runs **once**, with `USE_NONLONG_PRETRAIN = True` (default; needs 09 first). A second run
with it `False` is an optional ablation that shows whether the EyePACS pretraining helps.

## Any-image inference to PDR (`generate_progression.py`)
```bash
# --input can be one image or a folder; --start-stage N forces a known grade
python generate_progression.py --input ./my_fundus.jpg \
    --generator-checkpoint .../best-G.ckpt --classifier-checkpoint .../final_weights.pt \
    --seg-checkpoint EX=... --seg-checkpoint MA=... --seg-checkpoint HE=... --seg-checkpoint SE=... \
    --end-stage 4 --out-dir ./progression_outputs/
```
Per image: `stage<k>_{input,synth}.png`, `trajectory.png`, `trajectory.json` (Module 1 grade,
LBS and consistency per stage; Module 3 estimate if `--module3-checkpoint` is given).
`--dry-run` checks the plumbing with random weights.
