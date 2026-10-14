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

`stage_biomarker_profiles.json` holds, per stage, each biomarker's distribution and the lesions
present in >=50% of eyes (`characteristic_lesions`); per transition `k->k+1`, the biomarkers ranked
by effect size (`ranked_shifts`, Cohen's d + Mann-Whitney p) and the ones that barely move
(`shared_biomarkers`, |d| < 0.2). Caveat for the write-up: these come from Module 1's *predicted*
masks, so they describe what Module 1 sees per stage, not manual annotations.

Without `--module1-cache/--profiles`, step (c) still runs as plain stage translation (biomarker
losses off). `python train_module2_nonlong_pretrain.py --dry-run --save-dir /tmp/nl` smoke-tests the
whole pipeline on synthetic data. Compare `models_poc_from_nonlong/` against a from-scratch
`train_module2_poc.py` run with `evaluate_trajectory.py` to show whether pretraining helps.
