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
| 11 | `11_module1_drgnet_replication_colab` | **new** -- DRG-Net's own Module 1 runs, one model per dataset as in their repo: grading on FGADR (released split), IDRiD, EyePACS; lesion segmentation on FGADR and IDRiD (300 / 1000 ep by default; DRG-Net used 1500 / 2500), plus DDR (our addition, 250 ep, official train / val / test; `module1/dataprep/prepare_ddr_segmentation.py`); resumes from Drive checkpoints after a disconnect; test-set metrics summary | dataset zips on Drive (IDRiD "A. Segmentation" for IDRiD seg; a zip with DDR in its name for DDR seg) |
| 10 | `10_generate_progression_to_pdr_colab` | **new** -- any fundus image(s) -> Module 1 grade -> cascade to stage 4 (PDR), saves per-stage images/figure/JSON | 02, 03, 06 (07 optional) |

**`MODULE1_SOURCE`** (first code cell of 04_05 and 06-10, default `'drgnet'`) picks which Module 1
models the pipeline uses: `'drgnet'` = notebook 11's DRG-Net replication checkpoints (outputs go to
`_drgnet`-suffixed folders: `module1_cache_drgnet/`, `models_poc_tianjin_from_nonlong_drgnet/`, ...),
`'poc'` = notebooks 02/03's (the original folders). Use the same value in every notebook.

Order with the DRG-Net Module 1 (default):
**11 -> 04_05 -> 09 -> 06 -> 07 -> 08 -> 10** (01-03 not needed; 11 builds its own data files).

Order with the POC Module 1 (`MODULE1_SOURCE = 'poc'`):
**01 -> 02 + 03 -> 04_05 -> 09 -> 06 -> 07 -> 08 -> 10**.

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
