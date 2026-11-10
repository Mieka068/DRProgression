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
| 1.7 | EyePACS pickle: blank `Usage` = training image (was dropped -> 0 train images). EyePACS grader trained in nb 11 (capped at `EYEPACS_MAX_PER_GRADE` = 3000/grade, progressively balanced; test acc 0.762 / kappa 0.721). Pipeline default `CLASSIFIER_RUN = 'eyepacs'` (its `best_validation_weights.pt`); nb 04_05's grader-comparison cell (`module1/compare_graders_tianjin.py`) scores both graders against Tianjin's clinical grades to confirm the choice | `CLASSIFIER_RUN = 'fgadr'` |
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
| 7.1 | Laterality: disc found inside the field of view, pair constraint, validated on EyePACS (nb 11). EyePACS mixes mirrored and non-mirrored images (only ~61 % accuracy), so the validation is used only at >= 80 %; otherwise the default convention -- **check a few Tianjin images by eye and set `LATERALITY_DISC_LEFT_IS`** (nb 04_05/06/07). Written to `laterality_resolved_v2.csv` | `--legacy-estimator --no-pair-constraint` |
| 7.2 / 7.3 | `MIN_BASELINE_GRADE` (0 = keep Stage 0, 1 = >= mild NPDR); `tools/dataset_image_sizes.py` | -- |

**Checkpoints:** items 3.4/3.6/4.1/4.2 change the Generator/Discriminator `state_dict`s, so
checkpoints from `mod-2-NonLongIntegration` (including the EyePACS pretraining) do NOT load into the
default models -- the scripts stop with an error naming `--model-arch legacy`, which loads them. The
whole Module 2 chain (09 pretraining -> 06 fine-tuning) must be retrained on this branch.

**Re-run order:** 11 (tiled re-evaluation, 200-epoch segmenters, laterality validation on an
EyePACS sample) -> 04_05 (rebuild all Module 1 caches) -> 09 (EyePACS cache, profiles, pretraining) -> 06
(fine-tuning, trajectory evaluation, epoch-0 eval; then optionally the `USE_NONLONG_PRETRAIN = False`
ablation) -> 07 -> 08 -> 10.

**Module 1 caches** record the models and settings they were built with (`<cache>.meta.json`):
`apply_to_progression_data.py --resume` only continues a cache built the same way and otherwise
rebuilds it, so changing the grader or re-training the segmenters never mixes two Module 1
versions in one cache. **Segmentation resume** (nb 11) failed on PyTorch >= 2.6 before 9e0404e --
the segmenters in the "reran notebook 11" run are still the 100-epoch models; re-run nb 11.

# Round 4 (after the "3rd round run" notebooks in `outputs/`)
Pipeline outputs go to NEW Drive folders: `ROUND_TAG = '_r4'` (first code cell of 04_05-10) is
appended to every cache / Module 2 / Module 3 folder (`module1_cache_drgnet_comb_prefinals_r4/`, ...);
model folders (`drgnet_replication_prefinals/`) and the laterality file are shared. `ROUND_TAG = ''`
reads/writes round 3's folders again. Each change keeps the round-3 behaviour behind a flag:

| Item | Round-3 finding | Change (default now) | Round-3 behaviour |
|---|---|---|---|
| R4-1.1 | LongDR grade-0 eyes had LBS 0.1-0.34 (Tianjin/FIRE ~0.001): the segmenters ran on `FundusImagesNormalized`; 559 of 917 Module 2 pairs | LongDR's raw colour photos everywhere (`DRP_LONGDR_FOLDER`, `fundus_preprocess.py`); nb 04_05 shows both versions' lesion maps first and prints median LBS by grade per source | `DRP_LONGDR_FOLDER=FundusImagesNormalized` |
| R4-1.2 | 3:2 photos squashed into ovals at 128 px | Retina crop + pad to square in every Module 2/3 loader, evaluation and inference script (masks cropped with the same box); 256 px (`MODULE2_IMAGE_SIZE`) | `DRP_RETINA_CROP=0`, 128 |
| R4-1.3 | Segmentation LR never decayed (StepLR every 200 epochs); 100 -> 200 epochs halved IDRiD MA/SE AP | nb 11 `RUN['comb_seg_ft']`: 50-epoch fine-tune of each `comb_models_*` from its best weights, cosine LR 1e-3 -> 1e-5, image-only colour augmentation, Dice on MA + SE, model kept only if it beats the 200-epoch validation AP (`comb_ft_models_*`, used by 04_05-10 once all four exist); IDRiD stays external | `RUN['comb_seg_ft'] = False` |
| R4-1.4 | 16 `._*.jpg` files failed in 04_05; nb 11 figure cell crashed on a log without validation lines | skipped / figure skipped with a message | -- |
| R4-1.5 | Augmentation flipped/rotated the images but not the lesion mask; rotation filled grey | mask transformed with the images; black fill | -- |
| R4-2.5 | Trajectory evaluation ran on all Tianjin pairs, incl. the Generator's training patients | `evaluate_trajectory.py --exclude-train-from poc_results.json` (training patients listed in `split.train_tianjin_patients`); in-scope run (`--min-baseline-grade 1`); same for nb 08's figures | omit the flag |
| R4-2.6 | FID at n = 38 fell outside its own range | FID only from n >= 50, KID from n >= 10 (primary) | `--min-n-for-fid 5` |
| R4-3.1 | D's stage head: 17.5 % on real val images (balanced 27 %), yet 74-100 % "agreement" on fakes; cascade re-graded as the target 1.5 % | Frozen Module 1 grader (EyePACS) in G's loss, behind a random differentiable augmentation (`gan_guidance.FrozenGrader`), in pretraining and fine-tuning | `--lambda-grader 0` / no `--grader-loss-checkpoint` |
| R4-3.2 | 35 progressing training pairs; best epoch 11, D loss negative from epoch 34 | DiffAugment (color, translation, cutout) on every critic call + drift penalty 0.001 | `--diffaugment "" --lambda-drift 0` |
| R4-3.3 | Best epoch rewarded copying the baseline (most val pairs don't progress) | Selection on stage control (one stage above each held-out baseline) graded by the OTHER grader (FGADR) | `--selection-mode target`, same grader |
| R4-3.4 | -- | `tools/stage_control_check.py` (nb 09): pretrained G's outputs graded by both graders; `tools/grader_resolution_check.py` (nb 04_05): graders on real Tianjin images at 128/256 px = the cascade metric's ceiling | -- |
| R4-6 | Module 3 test AUROC 0.551, Brier = constant guess, heavy over-fitting | freeze 8 blocks; metrics also on in-scope eyes (baseline >= mild NPDR) with patient-bootstrap 95 % CIs; tabular baselines (grade event rate, logistic regression on grade + log LBS + Module 1 grade) on the same patients; output reported as P(progression by 2 years) | `--freeze-blocks 6 --report-min-baseline-grade 0 --n-bootstrap 0` |
| R4-7 | Registration net 45.0 px vs no warp 44.6 px; SIFT mean 1,738 px from a few wild matrices | `--method auto`: identity / net / SIFT (cropped frame, implausible affines rejected) scored on FIRE, cache built with the lowest median error | `REGISTRATION_METHOD = 'net'` |
| R4-scope | Demo used a grade-0 image | `generate_progression.py --min-start-stage 1` (nb 10) | `MIN_START_STAGE = 0` |

**Scope (baseline >= mild NPDR):** both Modules TRAIN on every eye (grade 0 eyes still teach image
realism and the 0 -> 1 transition, and the baseline grade is an input) but every reported metric has an
in-scope version (nb 06 `_in_scope.json`, nb 07 `in_scope`, nb 08 figures, nb 10 inputs).

**Re-run order (round 4):** 11 (fine-tune, ~1 h per lesion on an L4) -> 04_05 (caches + Phase-0
checks) -> 09 (EyePACS cache with the fine-tuned segmenters, 256 px pretraining with the grader loss,
stage-control check) -> 06 -> 07 -> 08 -> 10. Every 256 px Module 2 checkpoint is new (the 128 px ones
don't load at 256).

# Round 5 -- round 3's Module 2 settings + data fixes (branch `R5-Round3Baseline-Fixes`)
Round 4's Module 2 was worse than round 3's on the same 93 held-out pairs (PSNR 16.5 vs 24.0 dB, FID 176
vs 119, 0 % grade agreement every epoch, warped/blurry outputs with green/yellow marks at the disc and
fovea). Round 5 starts again from round 3's Module 2 setup and adds only the data fixes, so later
changes can be tested one at a time. Outputs go to `_r5` Drive folders (`ROUND_TAG = '_r5'` in
04_05-10); round 3's (no suffix) and round 4's (`_r4`) folders are never written to.

| Change | Why | Earlier behaviour |
|---|---|---|
| Text-safe retina crop: only the largest bright region counts as the retina (`fundus_preprocess.main_fov_region`, used by both `retina_box()`s and `resolve_eye_laterality.py`) | Tianjin's follow-ups print the ID, date and "R / Color" in the border; stacked in the same columns they passed the 2 % cut-off and stretched the crop box, so the follow-up retina filled ~60 % of the frame, off-centre (and SIFT rejected the >1.25x scale needed to align it) | `DRP_TEXT_SAFE_CROP=0` |
| SIFT pairs that fail are dropped from Module 2 training (cache value `None`, filtered in the FIRE/LongDR/Tianjin datasets); failure rate printed per source and saved as `registered_followups_sift_report.json` | Round 4 kept 455 of 917 failures unregistered, mixing aligned and unaligned targets. Over ~15 % failing: use `REGISTRATION_METHOD = 'identity'` (round 3's effective behaviour) | `train_registration.py --keep-sift-failures` |
| nb 06: `REGISTRATION_METHOD = 'sift'` | Best on FIRE in round 4 (11.4 px vs 44.2 px unaligned; round 3's network = no warp) | `'auto'` |
| 128 px (nb 06, 08, 09, 10) | Round 3's size; ~4x cheaper per epoch. 256 px only once 128 is clean | `MODULE2_IMAGE_SIZE = 'auto'` (256) |
| No grader loss (`LAMBDA_GRADER = 0`, nb 06 + 09), no DiffAugment / drift penalty (nb 06) | The grader loss reached CE 0.01 = G learned to fool it (the coloured marks); DiffAugment/drift were round-4 additions, off to keep round 3's setup | `1.0`, `'color,translation,cutout'`, `0.001` |
| Best epoch: val L1 - 0.1 x agreement with the target grade, graded by the EyePACS grader (`SELECTION_MODE = 'target'`) | Round 3's rule. Round 4's FGADR stage control stayed ~2 % all run (FGADR kappa on Tianjin ~0) | `'controllability'`, FGADR |
| nb 06 evaluation with no grader in the loss: primary runs graded by EyePACS (kappa 0.44 on Tianjin), the third by FGADR (`trajectory_eval_results_fgadr_grader.json`) | The grader with real agreement on Tianjin is the useful one; still only unseen patients (`--exclude-train-from`) | FGADR primary |
| nb 09: continues round 3's pretraining (copy of its `resume_state.pt` into the `_r5` folder) to 80 epochs; reuses round 4's 13k-image EyePACS cache (copied) | Round 3's pretraining was still improving at 40 epochs | `CONTINUE_ROUND3_PRETRAIN = False` / `REUSE_ROUND4_EYEPACS_CACHE = False` |
| nb 06 "Round 3 vs round 5" cell: round 3's best Generator and this round's through the same cascade evaluation on Tianjin patients NEITHER round could have trained on (`evaluate_trajectory.py --exclude-scope module2`, from the new `split.module2_tianjin_patients` in poc_results.json = every patient passing the pair-quality filter); round 3 also in its own frame (crop off). Tables + CSVs in `<run>/round3_comparison/` | Round 5's split differs from round 3's (SIFT drops change it), so round 5's "unseen" patients include some round 3 trained on | -- |
| Laterality file per round from round 5 on (`module1_cache<RUN_TAG>/`), computed with `LATERALITY_DISC_LEFT_IS = 'OS'` (round 3's hand-checked value) | The crop fix changes the frame the optic disc is located in | shared `module1_cache<RUN_SUFFIX>/` |

Crop check on the round-4 follow-up panels (old box vs text-safe box vs round 3's plain resize):
`docs/round5_crop_check.png`. Module 3 keeps round 4's `FREEZE_BLOCKS = 8`: round 4's nb 07 run no longer
over-fits (train/val NLL ~0.35 vs round 3's 0.07/0.34-0.73), and both rounds' test AUROC is at chance
(0.551 vs 0.515 [0.38, 0.67], 20 test events; the grade-only baseline scores 0.44 on the same patients).

Kept from round 4: LongDR colour photos, retina crop, round-4 segmenters (`comb_ft_models_*` once all
four exist), the larger EyePACS cache, the unseen-patient evaluation. Caveat: round 3's selection rule
mostly rewards copying the baseline (most held-out pairs don't progress) -- read the cascade/progression
numbers, not only PSNR/FID.

**Re-run order (round 5):** free Google Drive space (not round 3/4 folders) -> 04_05 (its retina-crop check
cell first, then caches + laterality with the crop fix) -> 09 (continues round 3's pretraining at 128 px) -> 06 (read the SIFT failure rate
first; its last evaluation cell compares with round 3) -> 07 -> 08 -> 10. Compare with round 3 on the unseen-patient evaluation. Next, one at a time:
`lambda_rec` 10 -> 5, then 256 px.

# Round 6 -- shape fixes + controls (branch `R6-ShapeFix-Controls`)
Round 5's fine-tuning made its own pretrained Generator worse on the same unseen Tianjin patients (step 1:
FID 64 -> 149, KID 0.027 -> 0.133, agreement 5.7 % -> 6.8 %), and its outputs turned square with edge bars
that grew along the cascade. Cause: with the retina filling the frame (round-5 crop), an enlarging SIFT warp
clips the follow-up's circle into a rounded square; the reconstruction L1 was unmasked and the critic saw
those targets as real. Round 6 changes only Module 2's fine-tuning and evaluation. It READS round 5's Module 1
caches, laterality file, SIFT cache, pretrained Generator and Module 3 as they are (`INPUT_ROUND_TAG = '_r5'`)
and WRITES only to `_r6` folders (`ROUND_TAG`). Rounds 3-5 are never written to.

| Change | Why | Earlier behaviour |
|---|---|---|
| `fov_mask.py` + `train_module2_poc.py --fov-mask`: G's output blacked out outside the baseline's retina circle; reconstruction L1 **and both critic inputs** restricted to baseline ∩ warped-follow-up retina | Nothing in training rewards clipped or black warp edges any more | unmasked |
| `--fov-mask` in `evaluate_trajectory.py`, `visualize_trajectory.py`, `generate_progression.py` (`synthesize_trajectory(..., fov_mask=)`): every cascade step locked to the REAL baseline's circle | Edge artifacts can't build up step to step; the circle can't shrink | raw output |
| nb 06: lr 1e-5 (G and D), 40 epochs / patience 10, pretrained critic (`--init-discriminator-mode full`; fall back to `reset-cls-head` if it won't load) | Don't wreck the pretrained Generator | 1e-4, 100 / 25, random critic |
| `--selection-mode kid`: best epoch = lowest KID vs the held-out real follow-ups (fixed subsets/seed); the starting Generator is scored as **epoch 0**, so if nothing beats it `best-G.ckpt` is the pretrained Generator; the copy-the-baseline KID is printed once as a reference | KID is unbiased at ~70 images; the agreement tiebreak would never apply | val L1 - 0.1 x agreement |
| `evaluate_trajectory.py --copy-baseline` and nb 06's five-row table on the same unseen patients (`--exclude-scope module2`): copy-the-baseline, round 3, round 5 pretrained, round 5 fine-tuned, round 6 -- all with `--fov-mask`, ending in a keep/drop-fine-tuning verdict. CSV in `<run>/five_way_comparison/` | A low FID/KID from copying only means "real photo" -- the floor every Generator must beat | round 3 vs round 5 cell |
| nb 08/10: `GENERATOR = 'finetuned' / 'pretrained'` (separate output folders); nb 10 `MIN_START_STAGE = 0` | Figures with whichever Generator the verdict picks; the grade-0 demo photo isn't skipped | fine-tuned only; 1 |

**Run order (round 6):** 06 -> 08 -> 10 (04_05, 09 and 07 not re-run). Not addressed yet: the ripple textures at
cascade steps 3-4 (compare `GENERATOR = 'pretrained'`), LongDR's 41 % SIFT failure rate, `lambda_rec` 10 -> 5, 256 px.

# Round 7 -- grade control + better evaluation (branch `R7-GradeControl-Eval`)
Round 6 fixed the shape (step-1 FID 121 -> 65, KID 0.096 -> 0.029 vs round 5 fine-tuned) but the Generator ignores the
requested grade: stage control 0.0 % in every epoch, best epoch 1 (essentially the pretrained Generator, KID tied with
it), and it doesn't beat copying the baseline (nb 06 verdict "NOT PROGRESSING YET"). The cascade still drifts grey/blurry
and grows thin bars at the circle's left/right edges. Round 7 again READS round 5's inputs (`INPUT_ROUND_TAG = '_r5'`)
and WRITES only to `_r7` folders; 04_05, 09 and 07 are not re-run.

| Change | Why | Earlier behaviour |
|---|---|---|
| nb 06: `LAMBDA_GRADER = 0.5` (frozen EyePACS grader back in G's loss), `LAMBDA_REC = 5` (new `--lambda-rec`), patience 15 | Nothing pushed G to change the grade; a strong L1 on mostly non-progressing pairs teaches copying. Half of round 4's weight: round 4 (1.0, lr 1e-4, random critic) fooled the loss grader 100 % at FID 266 | 0.0; 10 (fixed); 10 |
| `train_module2_poc.py --selection-mode control-kid`: eligible epochs need stage control >= `--selection-min-control` (0.2) AND KID <= `--selection-kid-slack` (2.0) x the pretrained Generator's (epoch 0); lowest KID among them; none -> best-G.ckpt = pretrained, said in the log. Stage control graded by the EyePACS grader; FGADR's logged each epoch (`--selection-check-classifier-checkpoint`). Loss plot gets a stage-control panel | KID alone rewards copying the baseline (round 6 picked epoch 1); the KID ceiling stops round 4's artefact epochs from winning. FGADR can't gate (kappa ~0 on Tianjin; round 4's FGADR stage control ~2 % all run) | `kid` |
| `evaluate_trajectory.py`: `final_step` block -- per pair's last image, exact agreement with the follow-up grade and **grade went up** (vs the grader's own reading of the real baseline), Wilson 95 % intervals, per step and pooled; `--second-classifier-checkpoint` (same numbers from FGADR); `--real-followup` (the ceiling row: grades the real follow-up photos) | Shows whether a grader can see real progression at all, and whether a gain is seen by a grader G wasn't trained against | exact agreement, one grader |
| nb 06 round-7 table (`<run>/round7_comparison/`): copy, real follow-up, round 5 pretrained, round 6, round 7, round 7 direct -- both graders; verdict separates "progressing", "loss grader only (artefacts)", "unconfirmed (FGADR blind)", "images degrade" | Round 7 trains against EyePACS, so EyePACS alone is optimistic | five-row table, one grader |
| `fov_mask.shrink_soften` / `match_color` / `--fov-erode 4 --fov-feather 3 --color-match` in evaluate/visualize/generate (nb 06/08/10); `--cascade-mode direct` (every stage from the real baseline) as a comparison row | Edge bars and grey drift along the cascade; direct vs cascade shows whether feeding G's output back in adds the blur | hard baseline mask, no colour match |

**Run order (round 7):** 06 -> 08 -> 10. Watch in nb 06: (1) the training log's stage control rising above 0 % (and
whether FGADR's check value follows); (2) the round-7 table's grade-up / agreement on both graders vs copy and the
pretrained row, read against the real-follow-up ceiling; (3) KID near the pretrained Generator's. If every epoch says
`not eligible (KID > ceiling)`, lower `LAMBDA_GRADER` to 0.3. Not addressed yet: residual Generator output (needs a new
nb 09 pretraining), 256 px, LongDR's SIFT failure rate.

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
