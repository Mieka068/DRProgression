"""
Per-cascade-step evaluation for Module 2's autoregressive multi-stage trajectory synthesis
(see DRForestGAN-v2/trajectory_inference.py).

Evaluates each cascade step against real follow-up images, bucketed by how many stages ahead
of the baseline that step is (step_count = followup_icdr - baseline_icdr). Also reports
Module 1's own re-grading consistency per step (does the synthesized image at step N read as
stage N to Module 1's own classifier?) -- a distinct signal from generator quality
(PSNR/SSIM/FID): a low consistency rate at a given step could mean the generator produced
something implausible, or that Module 1's own grader is unreliable on synthesized
(out-of-distribution) images. This script measures both without attempting to disentangle
which.

Consistency is reported two ways: a raw agreement rate (argmax prediction == target stage --
interpretable at a glance) and an AUC (roc_auc_score on Module 1's softmax score for the
target stage against whether the pair's real follow-up actually reached that stage --
directly comparable to DRForecastGAN's own published 0.87 internal / 0.85 external AUC for
this exact idea, which the raw rate is not since it discards ranking/confidence information).

Only Tianjin has a grade on both the baseline and the follow-up image (see
module3/dataset.py) -- FIRE and LongDR have no follow-up grade, so they cannot participate in
this evaluation. For a given pair, only the cascade step whose target stage equals the real
followup_icdr has a real image to compare against -- earlier/later steps in the same
trajectory have no ground truth for PSNR/SSIM/FID (Module 1's re-grading consistency check is
available at every step, since it only needs the synthesized image itself). This script
therefore reports one PSNR/SSIM comparison per pair, at that pair's own step_count bucket,
plus a consistency flag at every intermediate step along the way.

Round 7 additions (same JSON, new keys; nothing earlier changes):
  * "final_step": per step_count bucket (and "all" pooled), graded on each pair's LAST image (the
    one compared with the real follow-up): exact agreement with the real follow-up grade and
    "grade went up" = the grader rates it higher than it rates the real baseline (relative to the
    grader's own baseline reading, so a grader that over- or under-grades Tianjin overall doesn't
    count). The baseline is graded after the SAME retina masking / colour match as the synthesized
    image (finish_step), so the masking itself can't make the grade "go up" (eval_version 2; v1 graded
    the raw baseline and the masked rim alone moved EyePACS's grade up for 40 % of copied baselines);
    the real-follow-up row compares the two raw photos. Wilson 95% intervals -- n is small. With --second-classifier-checkpoint the same
    numbers from a second grader (round 7 trains G against the first one, so a gain only counts if
    the second grader sees it too). Same pairs for every row, unlike module1_consistency (whose
    step k pools every pair that passes through stage k).
  * --real-followup: no Generator -- the "last image" is the REAL follow-up photo. Its agreement is
    the ceiling for the agreement columns: how well each grader reads real Tianjin follow-ups.
    No PSNR/SSIM/FID (it would be the image against itself).
  * --fov-erode/--fov-feather/--color-match/--cascade-mode (fov_mask.add_cascade_args): the round 7
    cascade clean-up and the "direct from the baseline" control.
  * Grader resolution check (notebook 12), with --real-followup only: --include-stable also grades
    the pairs that did NOT progress (bucket "step_0", left out of "all"), so a grader's "grade went up"
    on progressed eyes can be read against its false-alarm rate on stable ones; --real-followup-native
    grades the original photo files (the grader's own 512 px preprocessing) instead of the
    --image-size tensors -- the full-resolution reference.

Usage (after a trained Module 2 checkpoint, a Tianjin module1 cache, and Module 1
classifier/segmentation checkpoints all exist):
    python evaluate_trajectory.py \
        --tianjin-dir /content/data/retinal-dr-longitudinal \
        --tianjin-module1-cache /content/drive/.../module1_outputs_tianjin.pt \
        --generator-checkpoint ./DRForestGAN-v2/stargan/models_poc/final-G.ckpt \
        --classifier-checkpoint /content/drive/.../final_weights.pt \
        --seg-checkpoint EX=/path/model_EX.pth.tar --seg-checkpoint MA=/path/model_MA.pth.tar \
        --out ./trajectory_eval_results.json

Dry run (random-init Module 1/2 weights, for smoke-testing the plumbing, no checkpoints needed):
    python evaluate_trajectory.py --tianjin-dir ... --tianjin-module1-cache ... \
        --dry-run --out /tmp/dry_run.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "DRForestGAN-v2"))
from base_model import Generator, add_arch_arg, arch_kwargs, check_run_config, load_checkpoint  # noqa: E402
from trajectory_inference import _module1_predict, _tensor_to_pil, synthesize_trajectory  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "module3"))
from dataset import TianjinSurvivalDataset  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "module1"))
from apply_to_progression_data import (  # noqa: E402
    ALL_LESIONS,
    CLASSIFIER_NORMS,
    add_classifier_norm_args,
    add_seg_inference_args,
    apply_classifier_norm_args,
    apply_seg_inference_args,
    load_classifier,
    load_segmentation_model,
    predict_grade,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tianjin_dataset import _module1_image_id_candidates  # noqa: E402
from fundus_preprocess import load_module2_image as _fp_load_module2_image  # noqa: E402
from fundus_preprocess import load_module2_mask as _fp_load_module2_mask  # noqa: E402
from fire_dataset import FIREDataset  # noqa: E402
from longdr_dataset import LongDRScreeningDataset  # noqa: E402
from image_quality_metrics import compute_fid_kid_stats  # noqa: E402
from fov_mask import CopyBaselineGenerator, add_cascade_args, cascade_kwargs, check_cascade_args, finish_step  # noqa: E402

# Bumped whenever this script's numbers change for the same inputs, so notebook 06 recomputes stale rows.
# 2 = round 7 fix: retina-mask edge where the circle touches the frame, and the baseline graded with the
#     same masking / colour match as the final image ("grade went up" like for like).
EVAL_VERSION = 2

SELF_CONSISTENCY_DATASETS = {"fire": FIREDataset, "longdr": LongDRScreeningDataset}
SELF_CONSISTENCY_TAG = "self-consistency -- not validated against ground truth"


def denorm(x):
    return ((x + 1) / 2).clamp_(0, 1)


def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n (sensible at small n and at 0 % / 100 %)."""
    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    lo = 0.0 if k == 0 else max(0.0, centre - half)  # exact at the ends (float error gives ~1e-17 otherwise)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return [float(lo), float(hi)]


def summarise_final_step(records, graders):
    """records: per-pair dicts {grader: (baseline_grade, final_grade, followup_grade)} -> rates + CIs."""
    out = {"n": len(records)}
    for g in graders:
        rows = [r[g] for r in records if g in r]
        n = len(rows)
        agree = sum(int(f == t) for _, f, t in rows)
        up = sum(int(f > b) for b, f, _ in rows)
        out[g] = {
            "n": n,
            "agree_rate": agree / n if n else None, "agree_ci95": wilson_interval(agree, n),
            "went_up_rate": up / n if n else None, "went_up_ci95": wilson_interval(up, n),
            "mean_grade_shift": float(np.mean([f - b for b, f, _ in rows])) if n else None,
        }
    return out


def load_module2_image(path, image_size):
    """[-1, 1] tensor of the image, retina-cropped like the training loaders (fundus_preprocess.py)."""
    return _fp_load_module2_image(path, image_size)


def load_mask(cache, baseline_path, image_size, dataset_dir=None):
    for candidate_id in _module1_image_id_candidates(baseline_path):
        if candidate_id in cache:
            # cropped with the image's own retina box -- needs the openable path (dataset_dir + relative)
            return _fp_load_module2_mask(cache[candidate_id]["mask"],
                                         os.path.join(dataset_dir or "", baseline_path), image_size)
    return torch.zeros(1, 1, image_size, image_size)


def load_module1_models(config, device):
    classifier = load_classifier(config.classifier_checkpoint, device=device, dry_run=config.dry_run)
    seg_models = {lesion: None for lesion in ALL_LESIONS}
    for entry in config.seg_checkpoint:
        lesion, path = entry.split("=", 1)
        seg_models[lesion] = load_segmentation_model(path, device=device, dry_run=config.dry_run)
    if config.dry_run and not config.seg_checkpoint:
        seg_models["EX"] = load_segmentation_model(None, device=device, dry_run=True)
    return classifier, seg_models


def evaluate_self_consistency(source_name, dataset_dir, G, classifier, seg_models, device, config):
    """
    FID/PSNR/SSIM for FIRE/LongDRScreening pairs, which have real baseline+follow-up images
    but NO real DR grade for either (see combined_dataset.py's module docstring). Since there
    is no ground-truth grade to synthesize a trajectory toward, this uses Module 1's own
    classifier prediction on the real follow-up image as the trajectory's target stage --
    i.e. it asks "if Module 1 itself thinks this pair progressed from stage X to stage Y, does
    the generator's stage-X-to-Y synthesis look like the real photo at stage Y?" That is a
    self-consistency check against Module 1's own (unverified) grading, not a validated
    comparison against ground truth the way the Tianjin path is -- every number this function
    returns must stay labeled with SELF_CONSISTENCY_TAG wherever it's shown.
    """
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    dataset_cls = SELF_CONSISTENCY_DATASETS[source_name]
    dataset = dataset_cls(dataset_dir, image_size=config.image_size)
    n_pairs = len(dataset)
    if config.self_consistency_max_pairs:
        n_pairs = min(n_pairs, config.self_consistency_max_pairs)

    per_step_psnr = defaultdict(list)
    per_step_ssim = defaultdict(list)
    per_step_images = defaultdict(lambda: {"real": [], "fake": []})
    n_skipped_no_progression = 0
    mask_size = config.image_size

    with torch.no_grad():
        for idx in range(n_pairs):
            sample = dataset[idx]
            baseline = sample["baseline"].unsqueeze(0).to(device)
            follow_up = sample["follow_up"].unsqueeze(0).to(device)

            baseline_stage, _, baseline_mask = _module1_predict(baseline, classifier, seg_models, device, mask_size)
            target_stage, _, _ = _module1_predict(follow_up, classifier, seg_models, device, mask_size)
            baseline_mask = baseline_mask.to(device)

            if target_stage <= baseline_stage:
                n_skipped_no_progression += 1
                continue  # Module 1 itself doesn't think this pair progressed -- nothing to evaluate

            trajectory = synthesize_trajectory(
                G, classifier, seg_models, baseline, baseline_mask,
                start_stage=baseline_stage, end_stage=target_stage, c_dim=config.c_dim, device=device,
                **cascade_kwargs(baseline, config),
            )
            synthesized = trajectory[-1]["image"]

            real_01 = denorm(follow_up)
            fake_01 = denorm(synthesized)
            step_count = target_stage - baseline_stage
            per_step_psnr[step_count].append(float(psnr_metric(fake_01, real_01)))
            per_step_ssim[step_count].append(float(ssim_metric(fake_01, real_01)))
            per_step_images[step_count]["real"].append(real_01.cpu())
            per_step_images[step_count]["fake"].append(fake_01.cpu())

    print(
        f"  [{source_name}] {n_pairs - n_skipped_no_progression}/{n_pairs} pairs used "
        f"(skipped {n_skipped_no_progression} Module-1-says-no-progression pairs)"
    )

    results = {}
    for step_count in sorted(per_step_psnr.keys()):
        n = len(per_step_psnr[step_count])
        entry = {
            "n_pairs": n,
            "mean_psnr": float(np.mean(per_step_psnr[step_count])),
            "mean_ssim": float(np.mean(per_step_ssim[step_count])),
            "note": SELF_CONSISTENCY_TAG,
        }
        fid_kid = compute_fid_kid_stats(
            per_step_images[step_count]["real"], per_step_images[step_count]["fake"], device,
            min_n_for_fid=config.min_n_for_fid, n_bootstrap=config.fid_n_bootstrap,
        )
        entry["fid"] = fid_kid["fid"]
        entry["fid_bootstrap_range"] = fid_kid["fid_bootstrap_range"]
        entry["fid_range_method"] = fid_kid["fid_range_method"]
        entry["kid_mean"] = fid_kid["kid_mean"]
        entry["kid_std"] = fid_kid["kid_std"]
        if fid_kid["note"]:
            entry["fid_note"] = fid_kid["note"]
        results[f"step_{step_count}"] = entry
        print(f"    [{source_name}] step_count={step_count}: n={n}, PSNR={entry['mean_psnr']:.3f}, "
              f"SSIM={entry['mean_ssim']:.3f}, FID={entry['fid']}, KID={entry['kid_mean']} "
              f"({SELF_CONSISTENCY_TAG})")
    return results


def evaluate(config):
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1/4] Loading Tianjin survival dataset (source of real baseline+follow-up grade pairs)...")
    survival_dataset = TianjinSurvivalDataset(
        config.tianjin_dir, module1_cache_path=config.tianjin_module1_cache, image_size=config.image_size,
        min_baseline_grade=config.min_baseline_grade,
    )
    manifest = survival_dataset.manifest
    held_out_info = None
    if config.exclude_train_from:
        # Round 4 (item R4-2.5): the Generator trained on the quality-filtered Tianjin pairs of
        # these patients -- evaluating on them would inflate every cascade number.
        _split = json.load(open(config.exclude_train_from)).get("split") or {}
        _key = "module2_tianjin_patients" if config.exclude_scope == "module2" else "train_tianjin_patients"
        _train = set(str(p) for p in _split.get(_key) or [])
        if config.exclude_scope == "module2" and not _train:
            raise SystemExit(f"{config.exclude_train_from} has no split.module2_tianjin_patients -- it comes from a "
                             "train_module2_poc.py run before round 5; pass round 5's poc_results.json.")
        if not _train:
            raise SystemExit(f"{config.exclude_train_from} has no split.train_tianjin_patients -- it comes from "
                             "a train_module2_poc.py run before round 4; re-train, or drop --exclude-train-from "
                             "(and report the numbers as including training patients).")
        before = len(manifest)
        manifest = manifest[~manifest["patient_id"].astype(str).isin(_train)].reset_index(drop=True)
        held_out_info = {"excluded_train_patients": len(_train), "exclude_scope": config.exclude_scope,
                         "pairs_before": before, "pairs_after": len(manifest)}
        print(f"  --exclude-train-from ({config.exclude_scope}): left out {len(_train)} "
              + ("Generator-training" if config.exclude_scope == "train" else "possible Module 2 training") + f" patients -> "
              f"{before} -> {len(manifest)} pairs ({manifest['patient_id'].nunique()} unseen patients)")
    cache = torch.load(config.tianjin_module1_cache, weights_only=False)

    print("\n[2/4] Loading Generator checkpoint and Module 1 classifier/segmentation models...")
    style_dim = config.style_dim or config.c_dim
    if config.real_followup:
        # Round 7 ceiling: no Generator -- each pair's last image is its REAL follow-up photo.
        G = None
        print("  --real-followup: no Generator -- grading the real follow-up photos (the agreement ceiling)")
    elif config.copy_baseline:
        # Round 6 control: every "synthesized" stage is the baseline photo unchanged. Same metrics,
        # same Module 1 re-grading -- the floor a real Generator has to beat.
        G = CopyBaselineGenerator().to(device)
        print("  --copy-baseline: no Generator -- every stage is the baseline image unchanged (control)")
    else:
        G = Generator(
            conv_dim=config.g_conv_dim, c_dim=config.c_dim, repeat_num=config.g_repeat_num, style_dim=style_dim,
            **arch_kwargs(config.model_arch)[0]).to(device)
        if not config.dry_run:
            check_run_config(config.generator_checkpoint, config.image_size, config.model_arch)
            load_checkpoint(G, config.generator_checkpoint, device, config.model_arch)
    if G is not None:
        G.eval()
    if config.fov_mask and G is not None:
        print("  --fov-mask: every synthesized step is blacked out outside the baseline's retina circle"
              + (f" (pulled in {config.fov_erode} px, edge faded over {config.fov_feather} px)"
                 if config.fov_erode or config.fov_feather else ""))
    if config.color_match and G is not None:
        print("  --color-match: every step's colours matched to the real baseline's")
    if config.cascade_mode == "direct" and G is not None:
        print("  --cascade-mode direct: every stage generated from the REAL baseline (control), not the previous step")
    classifier, seg_models = load_module1_models(config, device)
    # Graders for the round 7 final-step numbers: "primary" = --classifier-checkpoint (also the cascade's
    # re-grading grader), "second" = --second-classifier-checkpoint (optional).
    graders = {"primary": (classifier, None)}
    if config.second_classifier_checkpoint:
        graders["second"] = (load_classifier(config.second_classifier_checkpoint, device=device, dry_run=config.dry_run),
                             CLASSIFIER_NORMS[config.second_classifier_norm])
        print(f"  second grader: {config.second_classifier_checkpoint} ({config.second_classifier_norm})")

    def grade(name, img):
        """img: a [-1, 1] tensor (graded as the Module 2-sized image), or a file path (graded from the photo)."""
        model, norm = graders[name]
        mean, std = norm if norm else (None, None)
        src = img if isinstance(img, str) else _tensor_to_pil(img)
        return int(predict_grade(model, src, mean=mean, std=std, device=device))

    print(f"\n[3/4] Synthesizing trajectories for {len(manifest)} real pairs...")
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    per_step_psnr = defaultdict(list)
    per_step_ssim = defaultdict(list)
    per_step_images = defaultdict(lambda: {"real": [], "fake": []})
    consistency_by_step = defaultdict(list)  # step index within a trajectory (1, 2, ...) -> [bool]
    # AUC version of the same signal: score = Module 1's own softmax score for the target stage
    # (not just its argmax), label = whether this pair's real follow-up actually reached (>=)
    # that stage. roc_auc_score needs both classes present, so these are scored per-step below.
    consistency_scores_by_step = defaultdict(list)  # step index -> [(score, label)]
    final_step_records = defaultdict(list)  # step_count -> [{grader: (baseline, final, follow-up grade)}]
    n_skipped_no_progression = 0

    for _, row in manifest.iterrows():
        step_count = int(row["followup_icdr"]) - int(row["baseline_icdr"])
        keep_stable = G is None and config.include_stable and step_count == 0
        if step_count <= 0 and not keep_stable:
            n_skipped_no_progression += 1
            continue  # the cascade evaluates progression, not regression/no-change

        baseline_path = os.path.join(config.tianjin_dir, row["baseline_path"])
        followup_path = os.path.join(config.tianjin_dir, row["followup_path"])
        baseline_img = load_module2_image(baseline_path, config.image_size).to(device)
        real_followup_img = load_module2_image(followup_path, config.image_size).to(device)
        mask = load_mask(cache, row["baseline_path"], config.image_size, config.tianjin_dir)
        real_followup_stage = int(row["followup_icdr"])

        if G is None:  # --real-followup: grade the real photos only
            b_src, f_src = ((baseline_path, followup_path) if config.real_followup_native
                            else (baseline_img, real_followup_img))
            final_step_records[step_count].append(
                {g: (grade(g, b_src), grade(g, f_src), real_followup_stage) for g in graders})
            continue

        trajectory = synthesize_trajectory(
            G,
            classifier,
            seg_models,
            baseline_img,
            mask,
            start_stage=int(row["baseline_icdr"]),
            end_stage=int(row["followup_icdr"]),
            c_dim=config.c_dim,
            device=device,
            **cascade_kwargs(baseline_img, config),
        )
        for step_idx, entry in enumerate(trajectory[1:], start=1):
            if entry["consistency"] is not None:
                consistency_by_step[step_idx].append(bool(entry["consistency"]))
            if entry["grade_probs"] is not None:
                score = float(entry["grade_probs"][entry["stage"]])
                label = int(real_followup_stage >= entry["stage"])
                consistency_scores_by_step[step_idx].append((score, label))

        final_stage, synthesized = trajectory[-1]["stage"], trajectory[-1]["image"]
        assert final_stage == int(row["followup_icdr"])
        # baseline graded through the same masking / colour match as every synthesized step (like for like)
        _kw = cascade_kwargs(baseline_img, config)
        baseline_seen = finish_step(baseline_img, baseline_img, _kw["fov_mask"], _kw["color_match"])
        final_step_records[step_count].append(
            {g: (grade(g, baseline_seen), grade(g, synthesized), real_followup_stage) for g in graders})

        real_01 = denorm(real_followup_img)
        fake_01 = denorm(synthesized)
        psnr = float(psnr_metric(fake_01, real_01))
        ssim = float(ssim_metric(fake_01, real_01))

        per_step_psnr[step_count].append(psnr)
        per_step_ssim[step_count].append(ssim)
        per_step_images[step_count]["real"].append(real_01.cpu())
        per_step_images[step_count]["fake"].append(fake_01.cpu())

    print(
        f"  Skipped {n_skipped_no_progression} pairs with no real progression (step_count <= 0"
        + (", stable pairs kept as step_0" if config.include_stable and config.real_followup else "")
        + ") -- the cascade evaluates progression, not regression/no-change."
    )

    print("\n[4/4] Aggregating per-step metrics...")
    results = {"eval_version": EVAL_VERSION, "quality": {}, "module1_consistency": {},
               "evaluated_on": ("patients the Generator never trained on" if held_out_info
                                else "ALL Tianjin pairs, including the Generator's training patients (optimistic)"),
               "held_out": held_out_info, "min_baseline_grade": config.min_baseline_grade,
               "generator": ("real follow-up photos (ceiling)" if config.real_followup else
                             "copy-the-baseline control" if config.copy_baseline else config.generator_checkpoint),
               "fov_mask": bool(config.fov_mask), "fov_erode": config.fov_erode, "fov_feather": config.fov_feather,
               "color_match": bool(config.color_match), "cascade_mode": config.cascade_mode,
               "image_size": config.image_size, "include_stable": bool(config.include_stable),
               "real_followup_native": bool(config.real_followup_native),
               "graders": {"primary": config.classifier_checkpoint, "second": config.second_classifier_checkpoint,
                           "second_norm": config.second_classifier_norm if config.second_classifier_checkpoint else None},
               "final_step": {}}
    for step_count in sorted(per_step_psnr.keys()):
        n = len(per_step_psnr[step_count])
        entry = {
            "n_pairs": n,
            "mean_psnr": float(np.mean(per_step_psnr[step_count])),
            "mean_ssim": float(np.mean(per_step_ssim[step_count])),
        }
        # feature=2048 (standard Inception pool features), matching train_module2_poc.py's
        # convention -- see image_quality_metrics.py for why feature=64 is the wrong default
        # to compare against DRForecastGAN's published FID, and why a bootstrap range + KID
        # are reported alongside the point estimate at this sample size.
        fid_kid = compute_fid_kid_stats(
            per_step_images[step_count]["real"], per_step_images[step_count]["fake"], device,
            min_n_for_fid=config.min_n_for_fid, n_bootstrap=config.fid_n_bootstrap,
        )
        entry["fid"] = fid_kid["fid"]
        entry["fid_bootstrap_range"] = fid_kid["fid_bootstrap_range"]
        entry["fid_range_method"] = fid_kid["fid_range_method"]
        entry["kid_mean"] = fid_kid["kid_mean"]
        entry["kid_std"] = fid_kid["kid_std"]
        if fid_kid["note"]:
            entry["fid_note"] = fid_kid["note"]
        results["quality"][f"step_{step_count}"] = entry
        fid_range = entry["fid_bootstrap_range"]
        fid_range_str = f" (95% range [{fid_range[0]:.3f}, {fid_range[1]:.3f}])" if fid_range else ""
        print(
            f"  step_count={step_count}: n={n}, PSNR={entry['mean_psnr']:.3f}, "
            f"SSIM={entry['mean_ssim']:.3f}, FID={entry['fid']}{fid_range_str}, KID={entry['kid_mean']}"
        )

    # Round 7: final-step agreement / grade-went-up per step_count bucket and pooled, per grader.
    all_records = [r for step in sorted(final_step_records) if step > 0 for r in final_step_records[step]]
    for key, recs in [(f"step_{k}", final_step_records[k]) for k in sorted(final_step_records)] + [("all", all_records)]:
        entry = results["final_step"][key] = summarise_final_step(recs, list(graders))
        print(f"  final step [{key}] n={entry['n']}: " + " | ".join(
            f"{g}: agrees with follow-up grade {entry[g]['agree_rate']:.1%}, grade went up {entry[g]['went_up_rate']:.1%}"
            for g in graders if entry[g]["n"]))

    from sklearn.metrics import roc_auc_score

    for step_idx in sorted(consistency_by_step.keys()):
        flags = consistency_by_step[step_idx]
        rate = float(np.mean(flags))
        entry = {
            "n": len(flags),
            "module1_agrees_with_target_stage_rate": rate,
        }

        scored = consistency_scores_by_step.get(step_idx, [])
        labels = [label for _, label in scored]
        if scored and len(set(labels)) > 1:
            scores = [score for score, _ in scored]
            auc = float(roc_auc_score(labels, scores))
            entry["module1_consistency_auc"] = auc
            auc_str = f"{auc:.3f}"
        else:
            entry["module1_consistency_auc"] = None
            entry["module1_consistency_auc_note"] = (
                "only one class present among this step's real-followup-reached-target-stage "
                "labels -- AUC is undefined, omitted rather than reported misleadingly."
            )
            auc_str = "None"

        results["module1_consistency"][f"cascade_step_{step_idx}"] = entry
        print(f"  cascade step {step_idx}: Module 1 re-grading agreed with the target stage "
              f"{rate:.1%} of the time (n={len(flags)}); AUC={auc_str} "
              "(DRForecastGAN reports 0.87 internal / 0.85 external for this comparison)")

    if config.self_consistency_dirs and G is None:
        print("\n[extra] Self-consistency eval skipped -- --real-followup has no Generator to evaluate")
    elif config.self_consistency_dirs:
        print(f"\n[extra] Self-consistency eval ({SELF_CONSISTENCY_TAG}) for "
              f"{list(config.self_consistency_dirs.keys())}...")
        results["self_consistency"] = {"note": SELF_CONSISTENCY_TAG}
        for source_name, source_dir in config.self_consistency_dirs.items():
            results["self_consistency"][source_name] = evaluate_self_consistency(
                source_name, source_dir, G, classifier, seg_models, device, config
            )

    with open(config.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Wrote {config.out}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tianjin-dir", required=True)
    parser.add_argument("--tianjin-module1-cache", required=True)
    parser.add_argument("--generator-checkpoint", default=None, help="Required unless --dry-run")
    parser.add_argument("--classifier-checkpoint", default=None, help="Required unless --dry-run")
    parser.add_argument(
        "--seg-checkpoint",
        action="append",
        default=[],
        metavar="LESION=PATH",
        help="Repeatable, e.g. --seg-checkpoint EX=/path/model_EX.pth.tar --seg-checkpoint MA=/path/...",
    )
    parser.add_argument("--dry-run", action="store_true", help="Random-init Module 1 weights, no checkpoints needed")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--c-dim", type=int, default=5)
    parser.add_argument("--g-conv-dim", type=int, default=64)
    parser.add_argument("--g-repeat-num", type=int, default=6)
    parser.add_argument(
        "--style-dim", type=int, default=None, help="Defaults to c_dim, matching Generator's own default"
    )
    parser.add_argument("--min-n-for-fid", type=int, default=50,
                         help="FID only from this many pairs per step (biased below; was 5). KID from 10.")
    parser.add_argument("--fid-n-bootstrap", type=int, default=10,
                         help="Resamples (subsamples without replacement) for each step's FID range.")
    parser.add_argument(
        "--self-consistency-dirs",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        help="Repeatable. SOURCE is 'fire' or 'longdr'. Runs FID/PSNR/SSIM for that source's "
             "real pairs against the generator's synthesis, with the target stage taken from "
             "Module 1's own classifier prediction on the real follow-up image (neither "
             "source has a real grade) -- every number this produces is tagged "
             f"'{SELF_CONSISTENCY_TAG}' and must not be read alongside the Tianjin numbers as "
             "the same kind of claim. e.g. --self-consistency-dirs fire=./FIRE_dataset "
             "--self-consistency-dirs longdr=./LongDRScreening_20150209",
    )
    parser.add_argument("--self-consistency-max-pairs", type=int, default=200,
                         help="Cap on pairs evaluated per self-consistency source (0 = no cap).")
    parser.add_argument("--exclude-train-from", default=None,
                         help="train_module2_poc.py's poc_results.json: leave out every Tianjin patient the "
                              "Generator trained on (its split.train_tianjin_patients)")
    parser.add_argument("--exclude-scope", choices=["train", "module2"], default="train",
                         help="With --exclude-train-from: 'train' leaves out that run's training patients; "
                              "'module2' leaves out every patient that could reach Module 2 training in any round "
                              "(split.module2_tianjin_patients, round 5+) -- the same patients for every round's "
                              "Generator, for round-vs-round comparisons")
    parser.add_argument("--min-baseline-grade", type=int, default=0,
                         help="1 = only eyes starting at >= mild NPDR (the manuscript's scope)")
    parser.add_argument("--copy-baseline", action="store_true",
                         help="Round 6 control: no Generator, every stage = the baseline image unchanged")
    parser.add_argument("--fov-mask", action="store_true",
                         help="Round 6: black out each synthesized step outside the baseline's retina circle")
    add_cascade_args(parser)  # round 7: --fov-erode/--fov-feather/--color-match/--cascade-mode
    parser.add_argument("--real-followup", action="store_true",
                         help="Round 7 ceiling: no Generator, grade each pair's REAL follow-up photo")
    parser.add_argument("--include-stable", action="store_true",
                         help="With --real-followup: also grade pairs that did not progress (bucket step_0, not in 'all')")
    parser.add_argument("--real-followup-native", action="store_true",
                         help="With --real-followup: grade the original photo files instead of --image-size images")
    parser.add_argument("--second-classifier-checkpoint", default=None,
                         help="Round 7: a second grader for the final-step agreement / grade-went-up numbers")
    parser.add_argument("--second-classifier-norm", choices=list(CLASSIFIER_NORMS), default="fgadr",
                         help="Normalisation of --second-classifier-checkpoint")
    parser.add_argument("--out", default="./trajectory_eval_results.json")
    add_classifier_norm_args(parser)  # --classifier-norm eyepacs for an EyePACS-trained grader
    add_arch_arg(parser)  # must match the Generator checkpoint's run
    add_seg_inference_args(parser)
    args = parser.parse_args()
    check_cascade_args(parser, args)
    if args.real_followup and args.copy_baseline:
        parser.error("--real-followup and --copy-baseline are separate rows; pass one")
    if (args.include_stable or args.real_followup_native) and not args.real_followup:
        parser.error("--include-stable / --real-followup-native only apply with --real-followup")
    apply_classifier_norm_args(args)
    apply_seg_inference_args(args)

    if not args.dry_run and not args.copy_baseline and not args.real_followup and not args.generator_checkpoint:
        parser.error("--generator-checkpoint is required unless --dry-run is set")
    if not args.dry_run and not args.classifier_checkpoint:
        parser.error("--classifier-checkpoint is required unless --dry-run is set")

    self_consistency_dirs = {}
    for item in args.self_consistency_dirs:
        if "=" not in item:
            parser.error(f"--self-consistency-dirs must be SOURCE=PATH, got: {item}")
        source_name, source_dir = item.split("=", 1)
        if source_name not in SELF_CONSISTENCY_DATASETS:
            parser.error(f"Unknown self-consistency source '{source_name}', "
                         f"must be one of {list(SELF_CONSISTENCY_DATASETS.keys())}")
        self_consistency_dirs[source_name] = source_dir
    args.self_consistency_dirs = self_consistency_dirs

    evaluate(args)
